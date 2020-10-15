#=
Created on June 22, 2020
@author: Zhou Xiang
=#

using PyCall
@pyimport numpy as np

function pad_to_same_size(field, lower, upper)
    new_complex_amp = np.zeros([lower + upper, lower + upper],
                               dtype=np.complex)
    new_complex_amp[lower + 1:upper,
                    lower + 1:upper] = field[:complex_amp]
    return new_complex_amp
end