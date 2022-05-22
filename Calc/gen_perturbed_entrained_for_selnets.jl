using Distributed;ncores=5;addprocs(ncores)
using BenchmarkTools
using PyCall
using ProgressMeter
using NPZ
using JLD2
using FileIO
using DataFrames
using Printf
using ArgParse
using Images
using CSV
using Interact
using Plots;Plots.plotly()

@everywhere using ParallelDataTransfer
@everywhere using DifferentialEquations
@everywhere using LinearAlgebra
@everywhere using SparseArrays
@everywhere using Distributions
@everywhere using LsqFit
@everywhere using SharedArrays
@everywhere using StatsBase
@everywhere using Interpolations

# %%
Data_path = "/home/kh/문서/Data/scn_data" # folder containing perturb_nets.npz and back_groud.npz
np = pyimport("numpy")
dist_p = [1.7923754788410218, 3.7626200112845236, 3.7271376162162806, 925.897859609599]
Ex_md_dist = (dist_p[1]-1)/(dist_p[1]+dist_p[2]-2)*dist_p[end] + dist_p[3]
T_data = np.load(joinpath(Data_path,"perturb_nets.npz"),allow_pickle=true)
np_M = T_data.get("arr_0")

Back_data = np.load(joinpath(Data_path,"back_groud.npz"),allow_pickle=true)
const N_cell = Back_data.get("N_cell")[1]
function get_sparse(M::PyObject)::SparseMatrixCSC
    I,J = M.nonzero()
    data = M.data
    return sparse(I.+1,J.+1,data,N_cell,N_cell)
end


Y0 = 2*(rand(N_cell,2).-0.5)
tot_M = [get_sparse(np_M[s]) for s in 1:length(np_M)]
Diff_M = get_sparse(Back_data.get("Diff_M")[1])
const All_ind = Back_data.get("All_ind").+1
const Core_ind = Back_data.get("Core_ind").+1
const Shell_ind = Back_data.get("Shell_ind").+1
const sc_len = Back_data.get("sc_len")[1]
const val_Area = Back_data.get("val_Area")[1]
const P_size = Back_data.get("P_size")
Core_size = size(Core_ind)[1]
type_rand = truncated(Normal(24,2),0,Inf)
r_omega = 2pi./rand(type_rand,N_cell)
amp_factor = 0.3
type_unif = Normal(1*amp_factor,1.8*amp_factor)
r_mu = rand(type_unif,N_cell)
r_arg = hcat(r_mu,r_omega)
const Diff_Mᵀ = Diff_M'

sendto(workers(),N_cell=N_cell)
sendto(workers(),Core_size=Core_size)
sendto(workers(),P_size=P_size)
sendto(workers(),All_ind=All_ind)

@everywhere begin
    const S_net = zeros(N_cell)
    const D_x = zeros(N_cell)
    const D_y = zeros(N_cell)
    const Drift = zeros(N_cell)
    const L_v = vcat(ones(Core_size),zeros(N_cell-Core_size))


    function light_fun(x,light_p=12)
        if mod(x,24) < light_p
            1
        else
            0
        end
    end

    # model
    function MsimEq(dotY,Y,args,t)
        γ, DM, KM, mu, K, L_c, L_flag, arg_t, light_p = args
        x = @view Y[:,1]
        y = @view Y[:,2]
        dx = @view dotY[:,1]
        dy = @view dotY[:,2]
        r2 = @.sqrt(x^2 + y^2)
        mul!(S_net,K*KM,x)
        mul!(D_x,DM,x)
        mul!(D_y,DM,y)
        @. Drift = γ*(mu[:,1]-r2)
        if L_flag
            @. dx = Drift*x - mu[:,2]*y + D_x + S_net + L_c*L_v*light_fun(t - arg_t,light_p)
        else
            @. dx = Drift*x - mu[:,2]*y + D_x + S_net
        end
        @. dy = Drift*y + mu[:,2]*x + D_y
    end


    function ϕ(arr::Array{Float64,3})
        atan.(arr[:,2,:],arr[:,1,:])
    end
    ϕ(arr::Array{Float64,2}) = atan.(arr[:,2],arr[:,1])
    t = collect(0:1/6:999.9)

    # heat perturbation
    function affect_p!(integrator)
        amp = sqrt.(sum(x -> x^2,integrator.u,dims=2))
        integrator.u[:,1] = integrator.u[:,1] .- 0.67*amp
        integrator.u[:,2] = integrator.u[:,2] .- 0.67*amp
    end

    # callback function for entrainments
    function affect_pe!(integrator)
        if !integrator.p[7]
            integrator.p[7] = true
            integrator.p[8] = integrator.t
        end
    end

    function save_fun(u,t,integrator)
        phi = ϕ(integrator.u)
    end
    save_v = SavedValues(Float64,Array{Float64,1})
    s_cb = SavingCallback(save_fun,save_v,saveat=t[1]:1/2:t[end])

    function save_init(u,t,integrator)
        return copy(u)
    end
    save_v2 = SavedValues(Float64,Array{Float64,2})
    s_cb2 = SavingCallback(save_init,save_v2,saveat=[500.0])
    save_cb = CallbackSet(s_cb,s_cb2)


    # mean phase and synchrony
    function save_sig(u,t,integrator)
        phi = ϕ(integrator.u)
        cvv = exp.(phi*im)
        all_sync = abs(mean(cvv)) # synchrony in the all region
        all_core = abs(mean(cvv[1:Core_size])) # synchrony in the core region
        all_shell = abs(mean(cvv[Core_size+1:end])) # synchrony in the shell region
        mean_all = mean(phi)
        mean_core = mean(phi[1:Core_size])
        mean_shell = mean(phi[Core_size+1:end])
        std_phi = std(phi)
        return [all_sync,all_core,all_shell,mean_all,mean_core,mean_shell,std_phi]
    end
    save_v3 = SavedValues(Float64,Array{Float64,1})
    s_cb3 = SavingCallback(save_sig,save_v3,saveat=t[1]:1/2:t[end])
    s_cb_half3 =  SavingCallback(save_sig,save_v3,saveat=500.0:1/2:t[end])


    function cut_2pi(phases)
        if phases > 0
            (phases + pi) % (2 * pi) - pi
        else
            (phases - pi) % (2 * pi) + pi
        end
    end

    # convert time series to image seq
    function making_img(arr)
        t_size = size(arr)[2]
        img_seq = Array{Float32,3}(undef,P_size[1],P_size[2],t_size)
        img_seq = fill!(img_seq,NaN)
        cv_phi = @.(cos(arr)+1)/2
        Ind_arr = Array{CartesianIndex{2},1}(undef,N_cell)
        for (i,v) in enumerate(eachrow(cat(All_ind,dims=2)))
            Ind_arr[i] = CartesianIndex(v...)
        end
        for i=1:t_size
            tmp_img = @view img_seq[:,:,i]
            setindex!(tmp_img,cv_phi[:,i],Ind_arr)
        end
        return img_seq
    end

    # calculate time difference between two regions (core & shell)
    ht_arr = collect(1:24)
    function time_diff(t,A,B)
        new_t = LinRange(t[1],t[end],length(t)*3)
        intp_A = LinearInterpolation(t,A,extrapolation_bc = 0)
        intp_B = LinearInterpolation(t,B,extrapolation_bc = 0)
        new_A = intp_A(new_t)
        new_B = intp_B(new_t)
        dt = collect(LinRange(-new_t[end], new_t[end], 2*length(new_t)-1))
        tmp_idx =@. (dt > -12) & (dt < 12)
        xcorr = crosscor(new_B,new_A,StepRange(-size(new_A,1)+1,1,size(new_A,1)-1))
        return dt[tmp_idx][argmax(xcorr[tmp_idx])]
    end

    function etime_diff(t,A,B)
        new_t = LinRange(t[1],t[end],length(t)*3)
        intp_A = LinearInterpolation(t,A,extrapolation_bc = 0)
        intp_B = LinearInterpolation(t,B,extrapolation_bc = 0)
        new_A = intp_A(new_t)
        new_B = intp_B(new_t)
        dt = collect(LinRange(-new_t[end], new_t[end], 2*length(new_t)-1))
        xcorr = crosscor(new_B,new_A,StepRange(-size(new_A,1)+1,1,size(new_A,1)-1))
        return dt[argmax(xcorr)]
    end
end

# set base ode problem
K_range = LinRange(0,0.04,20)[2:2:end]
K = K_range[4]
D_coeff = 5.7/sc_len^2
t_end = 1000
dt = 1/6
p = [0.8,D_coeff*Diff_Mᵀ,SparseMatrixCSC(tot_M[1]'),r_arg,K,0.05,false,0,12]
prob = ODEProblem(MsimEq,Y0,(0.,t_end),p)
sendto(workers(),prob=prob)


split_idx = ncores
K_size = length(K_range)
M_size = length(tot_M)
split_idx = ncores
it_n = Int(ceil(M_size/split_idx))
tmp_i = [1+(i-1)*split_idx:1:min(i*split_idx,M_size) for i in 1:1:it_n]
f_name = [joinpath(Data_path,"perturb_imgs","perturb_img($s).jld2") for s = 1:it_n]
f_name_ent = joinpath(Data_path,"entrain_arrs.npz")
f_name_ent2 = joinpath(Data_path,"entrain_arrs_modi_lp.npz")
const jobs = RemoteChannel(()-> Channel{Tuple{Int64,SparseMatrixCSC{Float64,Int64}}}(ncores))
function put_jobs(n)
    for (i,m) in enumerate(tot_M[tmp_i[n]])
        put!(jobs,(tmp_i[n][i],m))
    end
end

# %% find PRC's unstalbe fixed point
const prc_data = RemoteChannel(()-> Channel{Tuple{Int64,Array{Float64,2},Float64,Float64,Array{Float64,2}}}(16))
@everywhere function get_prc(p,i,result)
    while true
        idx, M = take!(p)
        prob.p[3] = SparseMatrixCSC(M')
        sol_ref = solve(prob,BS3(),callback=s_cb,saveat=500.0,save_everystep=false)
        ref_phi = copy(hcat(save_v.saveval...))
        img_time = copy(save_v.t)
        y0_set = vcat(sol_ref.u[2])
        new_prob = remake(prob,u0=y0_set,tspan=(500.0,550.0))

        ht_arr = collect(1:27) .+ 500.
        prt_idx = Int.(collect(1:27)*2 .+ 1001)
        pphi = Array{Float64,1}(undef,length(ht_arr))
        dphi = Array{Float64,1}(undef,length(ht_arr))
        for (i,tt) in enumerate(ht_arr)
            pert_idx = prt_idx[i]
            pre_cb = PresetTimeCallback(tt,affect_p!)
            tmp_cb = CallbackSet(s_cb,pre_cb)
            sol = solve(new_prob,BS3(),callback=tmp_cb,save_everystep=false,tstop=tt)
            phi = copy(hcat(save_v.saveval...))
            post_phi = phi[:,pert_idx+1]
            pphi[i] = mean(ref_phi[:,pert_idx])
            dphi[i] = mean(post_phi)-pphi[i]
        end
        PRC = hcat(pphi,dphi)
        zero_idx = findall(@. dphi[2:end]*dphi[1:end-1] < 0)
        zero_idx = zero_idx[@. 0.3 < pphi[zero_idx] < 1]
        dp_pts = vcat([[ht_arr[s],dphi[s],ht_arr[s+1],dphi[s+1]]' for s in zero_idx]...)
        pp_pts = vcat([[pphi[s],dphi[s],pphi[s+1],dphi[s+1]]' for s in zero_idx]...)
        c,d,a,b = dp_pts[1,:]
        u_time = (-a*d+c*b)/(b-d)
        c1,d1,a1,b1 = pp_pts[1,:]
        upoints = (-a1*d1+c1*b1)/(b1-d1)
        put!(result,(idx,y0_set,u_time,upoints,PRC))
    end
end

ht_arr = collect(1:27) .+ 500.
new_y0_arr = Array{Float64,3}(undef,length(tot_M),N_cell,2)
PRC = Array{Float64,3}(undef,length(tot_M),length(ht_arr),2)
upoints = Array{Float64,1}(undef,length(tot_M))
utimes = Array{Float64,1}(undef,length(tot_M))

p_bar = Progress(it_n)
for i = 1:it_n
    st_t = time()
    @async put_jobs(i)
    for w in workers()
        remote_do(get_prc, w, jobs, i, prc_data)
    end
    tmp_result = [take!(prc_data) for _ in 1:size(collect(tmp_i[i]))[1]]
    args = [s[1] for s in tmp_result]
    new_y0_arr[tmp_i[i],:,:] = permutedims(cat([s[2] for s in tmp_result][sortperm(args)]...,dims=3),[3,1,2])
    utimes[tmp_i[i]] = [s[3] for s in tmp_result][sortperm(args)]
    upoints[tmp_i[i]] = [s[4] for s in tmp_result][sortperm(args)]
    PRC[tmp_i[i],:,:] = permutedims(cat([s[5] for s in tmp_result][sortperm(args)]...,dims=3),[3,1,2])
    ProgressMeter.next!(p_bar,showvalues=[(:iter,i),(:remain,it_n-i),(:dt,time()-st_t)])
end

begin
    save(joinpath(Data_path,"perturb_y0.jld2"),Dict("y0" => new_y0_arr,"prc" => PRC,
    "u_pts" => upoints, "Y0" => Y0, "r_arg" => r_arg, "utime" => utimes))
end
# %% entrain_y0
const ejobs = RemoteChannel(()-> Channel{Tuple{Int64,SparseMatrixCSC{Float64,Int64}}}(ncores))
function eput_jobs(n)
    for (i,m) in enumerate(tot_M[tmp_i[n]])
        put!(ejobs,(tmp_i[n][i],m))
    end
end


# const ent_data = RemoteChannel(()-> Channel{Tuple{Int64,Array{Float64,2},Float64,Array{Float64,1}}}(20))
const ent_data = RemoteChannel(()-> Channel{Tuple{Int64,Array{Float64,2},Array{Float64,2},Float64}}(ncores))
@everywhere function get_ent(p,i,result)
    while true
        idx, M = take!(p)
        prob.p[3] = SparseMatrixCSC(M')
        prob.p[7] = false
        tot_cb = CallbackSet(s_cb,s_cb3)
        sol_ref = solve(prob,BS3(),callback=tot_cb,saveat=500.0,save_everystep=true)
        ref_phi = copy(hcat(save_v.saveval...))
        result_arr = hcat(save_v3.saveval...)
        img_time = copy(save_v.t)
        y0_set = sol_ref(500)
        mean_rphi = dropdims(median(ref_phi,dims=1),dims=1)

        tmp_arr = mean_rphi[1001:end].- pi/2
        tmp_darr = diff(tmp_arr)
        tmp_t = save_v.t[1001:end]
        zero_idx = findall(@. tmp_arr[2:end]*tmp_arr[1:end-1] < 0)
        zero_idx = zero_idx[findfirst(tmp_darr[zero_idx] .> 0)]
        dp_pts = [tmp_t[zero_idx],tmp_arr[zero_idx],tmp_t[zero_idx+1],tmp_arr[zero_idx+1]]
        c,d,a,b = dp_pts
        u_time = (-a*d+c*b)/(b-d)

        put!(result,(idx,y0_set,result_arr,u_time))
    end
end

ht_arr = collect(1:12*2) .+ 500.
new_y0_arr = Array{Float64,3}(undef,length(tot_M),N_cell,2)
ref_arr = Array{Float16,3}(undef,length(tot_M),7,Int(1000/0.5))
utimes = Array{Float64,1}(undef,length(tot_M))

p_bar = Progress(it_n,showspeed=true)
st_t = time()
for i = 1:it_n
    @async eput_jobs(i)
    for w in workers()
        remote_do(get_ent, w, ejobs, i, ent_data)
    end
    tmp_result = [take!(ent_data) for _ in 1:size(collect(tmp_i[i]))[1]]
    args = [s[1] for s in tmp_result]
    new_y0_arr[tmp_i[i],:,:] = permutedims(cat([s[2] for s in tmp_result][sortperm(args)]...,dims=3),[3,1,2])
    ref_arr[tmp_i[i],:,:] = permutedims(cat([s[3] for s in tmp_result][sortperm(args)]...,dims=3),[3,1,2])
    utimes[tmp_i[i]] = [s[4] for s in tmp_result][sortperm(args)]
    ProgressMeter.next!(p_bar,showvalues=[(:iter,i),(:remain,it_n-i),(:elap_t,time()-st_t)])
    if i in [30,40,50,60,80,100,120,140]
        @everywhere GC.gc()
    end
end

begin
    save(joinpath(Data_path,"entrain_y0.jld2"),Dict("y0" => new_y0_arr,
    "Y0" => Y0, "r_arg" => r_arg, "utime" => utimes, "ref_arr" => ref_arr))
end
# %% generate perturbed data
jf = jldopen(joinpath(Data_path,"Perturb","perturb_y0.jld2"))
Y0 = jf["Y0"]
new_y0 = jf["y0"]
utimes = jf["utime"]
K_range = LinRange(0,0.04,20)[2:2:end]
K = K_range[4]
D_coeff = 5.7/sc_len^2
t_end = 1000
dt = 1/6
p = [0.8,D_coeff*Diff_Mᵀ,SparseMatrixCSC(tot_M[1]'),jf["r_arg"],K,true,0.,0.05,false]
prob = ODEProblem(MsimEq,Y0,(0.,t_end),p)
sendto(workers(),prob=prob)

const pₓ = RemoteChannel(()-> Channel{Tuple{Int64,SparseMatrixCSC{Float64,Int64},Array{Float64,2},Float64}}(20))
function put_jobs_My0(n)
    ti = collect(tmp_i[n])
    tmp_y0 = new_y0[ti,:,:]
    for (i,(m,i_t)) in enumerate(zip(tot_M[ti],utimes[ti]))
        put!(pₓ,(ti[i],m,tmp_y0[i,:,:],i_t))
    end
end

@everywhere function do_work_pt(p,i,result)
    while true
        idx,M,y0,ut = take!(p)
        new_prob = remake(prob,u0=y0,tspan=(500.0,1000.0))
        new_prob.p[3] = SparseMatrixCSC(M')
        new_prob.p[6] = true
        pre_cb = PresetTimeCallback(ut,new_affect_p!)
        tmp_cb = CallbackSet(s_cb_half,pre_cb)
        sol = solve(new_prob,BS3(),callback=tmp_cb,save_everystep=false,tstop=ut)
        tmp_phi = hcat(save_v.saveval...)
        result_arr = making_img(tmp_phi)
        put!(result,(idx,result_arr,ut))
    end
end


function perturb_attime()
    p_bar = Progress(it_n)
    for i = 1:it_n
        st_t = time()
        @async put_jobs_My0(i)
        for p in workers()
            remote_do(do_work_pt, p, pₓ, i, result)
        end
        tmp_result = [take!(result) for _ in 1:size(collect(tmp_i[i]))[1]]
        args = [s[1] for s in tmp_result]
        heat_time = [s[3] for s in tmp_result][sortperm(args)]
        img = cat([s[2] for s in tmp_result][sortperm(args)]...,dims=4)

        # img_data = cat(img_data,img,dims=4)
        ProgressMeter.next!(p_bar,showvalues=[(:iter,i),(:remain,it_n-i),(:dt,time()-st_t)])
        save(f_name[i],Dict("img_data" => img,"heat_time" => heat_time))
        img = nothing
        @everywhere GC.gc()
    end
end

perturb_attime()

# %% generate entrainment data
jf = jldopen(joinpath(Data_path,"entrain_y0.jld2"))
r_arg = jf["r_arg"]
Y0 = jf["Y0"]
new_y0 = jf["y0"]
utimes = jf["utime"]
K_range = LinRange(0,0.04,20)[2:2:end]
K = K_range[4]
D_coeff = 5.7/sc_len^2
t_end = 1000
dt = 1/6
p = [0.8,D_coeff*Diff_Mᵀ,SparseMatrixCSC(tot_M[1]'),r_arg,K,0.05,false,0,12]

prob = ODEProblem(MsimEq,Y0,(0.,t_end),p)
sendto(workers(),prob=prob)

const pₓ = RemoteChannel(()-> Channel{Tuple{Int64,SparseMatrixCSC{Float64,Int64},Array{Float64,2},Float64}}(20))
function put_jobs_My0(n)
    ti = collect(tmp_i[n])
    tmp_y0 = new_y0[ti,:,:]
    for (i,(m,i_t)) in enumerate(zip(tot_M[ti],utimes[ti]))
        put!(pₓ,(ti[i],m,tmp_y0[i,:,:],i_t))
    end
end


@everywhere function do_work_en(p,i,result)
    while true
        idx,M,y0,ut = take!(p)
        new_prob = remake(prob,u0=y0,tspan=(500.0,1000.0))
        new_prob.p[3] = SparseMatrixCSC(M')
        new_prob.p[7] = false

        pre_cb = PresetTimeCallback(ut,affect_pe!)
        tmp_cb = CallbackSet(s_cb_half3,pre_cb)

        sol = solve(new_prob,BS3(),callback=tmp_cb,save_everystep=false,tstop=ut)
        ent_result = hcat(save_v3.saveval...)
        put!(result,(idx,ent_result,new_prob.p[8]))
    end
end



function entrain_attime()
    p_bar = Progress(it_n)
    ent_arr = Array{Float16,3}(undef,7,Int(500/0.5),0)
    s_time = Array{Float16,1}(undef,0)
    for i = 1:it_n
        st_t = time()
        @async put_jobs_My0(i)
        for p in workers()
            remote_do(do_work_en, p, pₓ, i, result_ent)
        end
        tmp_result = [take!(result_ent) for _ in 1:size(collect(tmp_i[i]))[1]]
        args = [s[1] for s in tmp_result]
        tmp_arrs = cat([s[2] for s in tmp_result][sortperm(args)]...,dims=3)
        append!(s_time, [s[3] for s in tmp_result][sortperm(args)])
        ent_arr = cat(ent_arr,tmp_arrs,dims=3)
        # img_data = cat(img_data,img,dims=4)
        ProgressMeter.next!(p_bar,showvalues=[(:iter,i),(:remain,it_n-i),(:dt,time()-st_t)])
    end
    npzwrite(f_name_ent,ent_arr=ent_arr, s_time=s_time)
end

entrain_attime()
