#=
Created on June 22, 2020
@author: Zhou Xiang
=#

using Images

function pad_to_same_size(complex_amp::Array{Complex{Float64}, 2},
                          lower::Int64,
                          upper::Int64)
    new_complex_amp = zeros(ComplexF64, lower + upper, lower + upper)
    new_complex_amp[lower + 1:upper, lower + 1:upper] = complex_amp
    return new_complex_amp
end

function resize_to_same_size(complex_amp_part::Array{Float64,2},
                             median_N::Int64)
    new_size = (median_N, median_N)
    resized_complex_amp_part = imresize(complex_amp_part, new_size)
    return resized_complex_amp_part
end