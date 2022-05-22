using Distributed;ncores=16;addprocs(ncores)
using BenchmarkTools
using PyCall
using ParallelDataTransfer
using ProgressMeter
using NPZ
using JLD2
using FileIO
using DataFrames
using Printf
using ArgParse

@everywhere using DifferentialEquations
@everywhere using LinearAlgebra
@everywhere using SparseArrays
@everywhere using Distributions
@everywhere using LsqFit
@everywhere using SharedArrays


function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--deg_r", "-r"
            help = "degree relation"
            arg_type = String
            required = true
        "--deg_d", "-d"
            help = "degree of degree relation"
            arg_type = Int
            required = true
    end
    return parse_args(s)
end


Data_path = "/mnt/DataDrive/Python_codes/SCN_modeling"
np = pyimport("numpy")
arg_p = parse_commandline()
deg_r = arg_p["deg_r"]
deg_d = arg_p["deg_d"]
println("$deg_r($deg_d) start!")

dist_p = [1.7923754788410218, 3.7626200112845236, 3.7271376162162806, 925.897859609599]
Ex_md_dist = (dist_p[1]-1)/(dist_p[1]+dist_p[2]-2)*dist_p[end] + dist_p[3]
T_data = np.load(joinpath(Data_path,"Nets","Adjacency_all_$deg_r($deg_d).npz"),allow_pickle=true)
c = T_data.get("clust_data")
l = T_data.get("path_length")
r = T_data.get("assort_data")
e1 = T_data.get("md_dist") .- Ex_md_dist
e2 = T_data.get("ex2_err")
inps = T_data.get("sel_inps")
dd = replace(Array{String,1}(inps[:,5]),"p" => 1,"i"=>-1)
ds = replace(Array{String,1}(inps[:,6]),"in" => 1,"tot" => 0,"out"=>-1)
dr = Array{Float64,1}(inps[:,end])

if deg_r == "prop"
    deg_arr = fill!(Array{Int8,1}(undef,length(c)),deg_d)
elseif deg_r == "inv"
    deg_arr = fill!(Array{Int8,1}(undef,length(c)),-deg_d)
else
    deg_arr = fill!(Array{Int8,1}(undef,length(c)),0)
end
Back_data = np.load(joinpath(Data_path,"back_groud.npz"),allow_pickle=true)

np_tot_M = T_data.get("tot_M")
const N_cell = Back_data.get("N_cell")[1]
function get_sparse(M::PyObject)::SparseMatrixCSC
    I,J = M.nonzero()
    data = M.data
    return sparse(I.+1,J.+1,data,N_cell,N_cell)
end

Y0 = 2*(rand(N_cell,2).-0.5)
tot_M = [get_sparse(np_tot_M[s]) for s in 1:length(np_tot_M)]
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

function calc_Conn_stren(ndarr)
    ne_cc = sum(ndarr[1:Core_size,1:Core_size])
    ne_ss = sum(ndarr[Core_size+1:end,Core_size+1:end])
    ne_cs = sum(ndarr[1:Core_size,Core_size+1:end])
    ne_sc = sum(ndarr[Core_size+1:end,1:Core_size])
    return Int64.([ne_cc,ne_ss,ne_cs,ne_sc])
end
noe = hcat([calc_Conn_stren(s) for s in tot_M]...)

tmp_df = DataFrame(c=c,l=l,r_ii=r[:,1],r_oo=r[:,2],r_io=r[:,3],r_oi=r[:,4],
            cc=noe[1,:],ss=noe[2,:],cs=noe[3,:],sc=noe[4,:],e1=e1,e2=e2,ds=ds,
            dr=dr,dd=dd,deg_d = deg_arr)


sendto(workers(),N_cell=N_cell)
sendto(workers(),Core_size=Core_size)
sendto(workers(),P_size=P_size)
sendto(workers(),All_ind=All_ind)

@everywhere begin
    const S_net = zeros(N_cell)
    const D_x = zeros(N_cell)
    const D_y = zeros(N_cell)
    const Drift = zeros(N_cell)
    function MsimEq(dotY,Y,args,t)
        γ, DM, KM, mu, K = args
        x = @view Y[:,1]
        y = @view Y[:,2]
        dx = @view dotY[:,1]
        dy = @view dotY[:,2]
        r2 = @.sqrt(x^2 + y^2)
        mul!(S_net,K*KM,x)
        mul!(D_x,DM,x)
        mul!(D_y,DM,y)
        @. Drift = γ*(mu[:,1]-r2)
        @. dx = Drift*x - mu[:,2]*y + D_x + S_net
        @. dy = Drift*y + mu[:,2]*x + D_y
    end
    function ϕ(arr::Array{Float64,3})
        atan.(arr[:,2,:],arr[:,1,:])
    end
    ϕ(arr::Array{Float64,2}) = atan.(arr[:,2],arr[:,1])
    t = collect(0:1/6:999.9)
    function save_vals(u,t,integrator)
        phi = ϕ(integrator.u)
        cvv = exp.(phi*im)
        all_sync = abs(mean(cvv))
        all_core = abs(mean(cvv[1:Core_size]))
        all_shell = abs(mean(cvv[Core_size+1:end]))
        mean_all = mean(phi)
        mean_core = mean(phi[1:Core_size])
        mean_shell = mean(phi[Core_size+1:end])
        std_phi = std(phi)
        return [all_sync,all_core,all_shell,mean_all,mean_core,mean_shell,std_phi]
    end
    save_v = SavedValues(Float64,Array{Float64,1})
    save_tv = SavedValues(Float64,Array{Float64,1})
    s_cb = SavingCallback(save_vals,save_v,saveat=t[1]:1/2:t[end])
    s_cb_ttx = SavingCallback(save_vals,save_tv,saveat=500:1/2:t[end])

    function save_y0(u,t,integrator)
        return integrator.u
    end

    save_v2 = SavedValues(Float64,Array{Float64,2})
    s_cb2 = SavingCallback(save_y0,save_v2,saveat=[500])

    function affect!(integrator)
        if integrator.p[5] != 0
            integrator.p[3] = integrator.p[3]*integrator.p[5]
            integrator.p[5] = 0
        else
            integrator.p[5] = 1.
        end
    end

    p_cb = PresetTimeCallback([500,500+144],affect!)
    stb_cb = CallbackSet(s_cb,s_cb2)
    ttx_cb = CallbackSet(s_cb_ttx,p_cb)
end

K = 0.0
D_coeff = 5.7/sc_len^2
p = [0.8,D_coeff*Diff_Mᵀ,SparseMatrixCSC(tot_M[1]'),r_arg,K,true]
t_end = 1000
dt = 1/6
prob = ODEProblem(MsimEq,Y0,(0.,t_end),p)
sendto(workers(),prob=prob)

K_range = LinRange(0,0.04,20)[2:2:end]
split_idx = ncores
K_size = length(K_range)
M_size = length(tot_M)
const pₐ = RemoteChannel(()-> Channel{Tuple{Int64,Float64,SparseMatrixCSC{Float64,Int64}}}(20))
const result =  RemoteChannel(()-> Channel{Tuple{Array{Float64,2},Array{Float64,2},Int}}(20))

function main()
    split_idx = ncores
    it_n = Int(ceil(M_size/split_idx))
    tmp_i = [1+(i-1)*split_idx:1:min(i*split_idx,M_size) for i in 1:1:it_n]

    function put_jobs_M(n,k)
        for (i,m) in enumerate(tot_M[tmp_i[n]])
            put!(pₐ,(tmp_i[n][i],k,m))
        end
    end

    @everywhere function do_work_M(p,i,result)
        while true
            idx,K,M = take!(p)
            prob.p[5] = K
            prob.p[3] = SparseMatrixCSC(M')
            stable_sol = solve(prob,BS3(),save_everystep=false,callback=stb_cb)
            st_result = hcat(save_v.saveval...)
            result_arr1 = [save_v.t';st_result]
            nprob = remake(prob,u0=save_v2.saveval[1],t=(500.,t[end]))
            ttx_sol = solve(nprob,BS3(),save_everystep=false,callback=ttx_cb)
            ttx_result = hcat(save_tv.saveval...)
            result_arr2 = [save_tv.t';ttx_result]
            put!(result,(result_arr1,result_arr2,idx))
        end
    end

    itK_range = K_range
    n_len = length(tmp_i)*size(itK_range)[1]
    p_bar = Progress(n_len)
    jj = 0
    for (j,K) in enumerate(itK_range)
        K_val_str = @sprintf "%.1E" K
        p_folder = "/mnt/DataDrive/Python_codes/SCN_modeling/patterns($(deg_r[1])$deg_d)"
        if ~isdir(p_folder)
            mkpath(p_folder)
        end
        jj = j
        filenames_arr = joinpath(p_folder,"K_($K_val_str).jld2")
        if isfile(filenames_arr)
            continue
        end
        result_st = Array{Float16,3}(undef,8,Int(1000/0.5),0)
        result_ttx = Array{Float16,3}(undef,8,Int(500/0.5),0)
        for i = 1:it_n
            st_t = time()
            @async put_jobs_M(i,K)
            for p in workers()
                remote_do(do_work_M, p, pₐ, i, result)
            end
            tmp_result = [take!(result) for _ in 1:size(collect(tmp_i[i]))[1]]
            args = [s[3] for s in tmp_result]
            arr_st = cat([s[1] for s in tmp_result][sortperm(args)]...,dims=3)
            arr_ttx = cat([s[2] for s in tmp_result][sortperm(args)]...,dims=3)
            result_st = cat(result_st,arr_st,dims=3)
            result_ttx = cat(result_ttx,arr_ttx,dims=3)
            p_idx = split_idx*(jj-1) + i
            ProgressMeter.next!(p_bar,showvalues=[(:iter,p_idx),(:remain,n_len-p_idx),(:dt,time()-st_t)])
        end
        save(filenames_arr,Dict("stable_arr" => result_st,"ttx_arr" => result_ttx))
        @everywhere GC.gc()
    end
end
main()
