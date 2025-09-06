using GenX
using Gurobi
# current_dir = dirname(@__FILE__)
# println(current_dir)
# run_genx_case!(dirname(@__FILE__))
run_genx_case!(dirname(@__FILE__),Gurobi.Optimizer)