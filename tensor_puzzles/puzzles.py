from lib import draw_examples, make_test, run_test
import torch
import numpy as np
from torchtyping import TensorType as TT
tensor = torch.tensor

# NOTES
# Things you can do
# a * b, a + b etc. tries to the operation compotentwise, tensoring up with a vector of ones if needed
# a == b or a > b etc. does the comparison (possibly tensoring up) and returns a tensor of booleans
# arange, where as below
# a[..., None], a[None, ...], tensors up
# if v has booleans, a[v] returns the subtensor where v has True
# a[[3,2,1, 0]] gives the 3rd, 2nd, 1st, 0th rows in that order (can also repeat, or do a[:,[1,2,3]] for columns); see also a[arange(2)] for the first 2 rows
# a[[0,1,2],[5,6,7]] takes the entries in (0,5), (1,6), (2,7) and puts it in a vector
# This can probably? be generalized to higher tensor rank




def arange(i: int):
    "Use this function to replace a for-loop."
    return torch.tensor(range(i))

#draw_examples("arange", [{"" : arange(i)} for i in [5, 3, 9]])

# Example of broadcasting.
#examples = [(arange(4), arange(5)[:, None]) ,
#            (arange(3)[:, None], arange(2))]
#draw_examples("broadcast", [{"a": a, "b":b, "ret": a + b} for a, b in examples])

def where(q, a, b):
    "Use this function to replace an if-statement."
    return (q * a) + (~q) * b

# In diagrams, orange is positive/True, where is zero/False, and blue is negative.

#examples = [(tensor([False]), tensor([10]), tensor([0])),
#            (tensor([False, True]), tensor([1, 1]), tensor([-10, 0])),
#            (tensor([False, True]), tensor([1]), tensor([-10, 0])),
#            (tensor([[False, True], [True, False]]), tensor([1]), tensor([-10, 0])),
#            (tensor([[False, True], [True, False]]), tensor([[0], [10]]), tensor([-10, 0])),
#           ]
#draw_examples("where", [{"q": q, "a":a, "b":b, "ret": where(q, a, b)} for q, a, b in examples])


### PUZZLE 1

def ones_spec(out):
    for i in range(len(out)):
        out[i] = 1
        
def ones(i: int) -> TT["i"]:
    return arange(i) * 0 + 1

#test_ones = make_test("one", ones, ones_spec, add_sizes=["i"])
#run_test(test_ones)


### PUZZLE 2

def sum_spec(a, out):
    out[0] = 0
    for i in range(len(a)):
        out[0] += a[i]
        

def sum(a: TT["i"]) -> TT[1]:
    return ones(a.shape[0])[None, :] @ a
   
# Previously:
#return a @ ones(a.shape[0])[:,None]
# This doesn't work when a is multidimensional tensor


#test_sum = make_test("sum", sum, sum_spec)
#run_test(test_sum)


### PUZZLE 3

def outer_spec(a, b, out):
    for i in range(len(out)):
        for j in range(len(out[0])):
            out[i][j] = a[i] * b[j]
            
def outer(a: TT["i"], b: TT["j"]) -> TT["i", "j"]:
    return a[:,None] @ b[None, :]

#test_outer = make_test("outer", outer, outer_spec)
#run_test(test_outer)



### PUZZLE 4

def diag_spec(a, out):
    for i in range(len(a)):
        out[i] = a[i][i]
        
def diag(a: TT["i", "i"]) -> TT["i"]:
    return a[arange(a.shape[0]), arange(a.shape[0])]


#test_diag = make_test("diag", diag, diag_spec)
#run_test(test_diag)


### PUZZLE 5

def eye_spec(out):
    for i in range(len(out)):
        out[i][i] = 1
        
def eye(j: int) -> TT["j", "j"]:
    return where((arange(j)[:, None] - arange(j)[None, :]) == 0, 1, 0)
 
    
#test_eye = make_test("eye", eye, eye_spec, add_sizes=["j"])
#run_test(test_eye)


### PUZZLE 6

def triu_spec(out):
    for i in range(len(out)):
        for j in range(len(out)):
            if i <= j:
                out[i][j] = 1
            else:
                out[i][j] = 0
                
def triu(j: int) -> TT["j", "j"]:
    return where((arange(j)[:, None] - arange(j)[None, :]) <= 0, 1, 0)


#test_triu = make_test("triu", triu, triu_spec, add_sizes=["j"])
#run_test(test_triu)


### PUZZLE 7

def cumsum_spec(a, out):
    total = 0
    for i in range(len(out)):
        out[i] = total + a[i]
        total += a[i]

def cumsum(a: TT["i"]) -> TT["i"]:
    return sum(where((arange(a.shape[0])[:, None] - arange(a.shape[0])[None, :]) >= 0, 1, 0) * a)[:,0]		


# Make a lower triangular matrix of 1s, then multiply, then sum
# This might not be the best way


#test_cumsum = make_test("cumsum", cumsum, cumsum_spec)
#run_test(test_cumsum)



### PUZZLE 8

def diff_spec(a, out):
    out[0] = a[0]
    for i in range(1, len(out)):
        out[i] = a[i] - a[i - 1]

def diff(a: TT["i"], i: int) -> TT["i"]:
    return a - a[where(arange(a.shape[0]) > 0, arange(a.shape[0]) - 1, 0)] + a * where(arange(a.shape[0]) == 0, 1, 0)

#test_diff = make_test("diff", diff, diff_spec, add_sizes=["i"])
#run_test(test_diff)





## PUZZLE 9

def vstack_spec(a, b, out):
    for i in range(len(out[0])):
        out[0][i] = a[i]
        out[1][i] = b[i]

def vstack(a: TT["i"], b: TT["i"]) -> TT[2, "i"]:
    return where(arange(2) == 0, 1, 0)[:, None] * a + where(arange(2) == 1, 1, 0)[:, None] * b


#test_vstack = make_test("vstack", vstack, vstack_spec)
#run_test(test_vstack)

# PUZZLE 10

def roll_spec(a, out):
    for i in range(len(out)):
        if i + 1 < len(out):
            out[i] = a[i + 1]
        else:
            out[i] = a[i + 1 - len(out)]
            
def roll(a: TT["i"], i: int) -> TT["i"]:
    return a[where(arange(i) + 1 == i, 0, arange(i) + 1)]


#test_roll = make_test("roll", roll, roll_spec, add_sizes=["i"])
#run_test(test_roll)



# PUZZLE 11

def flip_spec(a, out):
    for i in range(len(out)):
        out[i] = a[len(out) - i - 1]
        
def flip(a: TT["i"], i: int) -> TT["i"]:
    return a[i - 1 - arange(i)]

#test_flip = make_test("flip", flip, flip_spec, add_sizes=["i"])
#run_test(test_flip)


# PUZZLE 12

def compress_spec(g, v, out):
    j = 0
    for i in range(len(g)):
        if g[i]:
            out[j] = v[i]
            j += 1


# First, take v[g] which takes only the entries corresponding to True.  The result is a smaller vector
# Then, use the idea from puzzle 13 to add the zeros back
# Note: v[g] has shape (i,), so it automatically makes it a column vector
def compress(g: TT["i", bool], v: TT["i"], i:int) -> TT["i"]:
    return eye(v.shape[0])[:, 0:(v[g].shape[0])] @ v[g]


#test_compress = make_test("compress", compress, compress_spec, add_sizes=["i"])
#run_test(test_compress)

# PUZZLE 13

def pad_to_spec(a, out):
    for i in range(min(len(out), len(a))):
        out[i] = a[i]

# Take the identity matrix truncated to shape (j, i) and then multiply it with v which has shape (i, )        
def pad_to(a: TT["i"], i: int, j: int) -> TT["j"]:
    return eye(max(i, j))[0:j, 0:i] @ a


#test_pad_to = make_test("pad_to", pad_to, pad_to_spec, add_sizes=["i", "j"])
#run_test(test_pad_to)

# PUZZLE 14

def sequence_mask_spec(values, length, out):
    for i in range(len(out)):
        for j in range(len(out[0])):
            if j < length[i]:
                out[i][j] = values[i][j]
            else:
                out[i][j] = 0


# First, make a matrix arange(j)[None, :] whose columns are 0, 1, 2, etc.
# Then, compare it with the length viewed as a column vector tensor a row vector of ones
# Take either the value, or 0 depending on the comparison result

def sequence_mask(values: TT["i", "j"], length: TT["i", int]) -> TT["i", "j"]:
    return where(arange(values.shape[1])[None, :] * ones(length.shape[0])[:, None] < length[:,None], values, 0)


def constraint_set_length(d):
    d["length"] = d["length"] % d["values"].shape[1]
    return d


#test_sequence = make_test("sequence_mask",
#    sequence_mask, sequence_mask_spec, constraint=constraint_set_length
#)
#run_test(test_sequence)



# PUZZLE 15

def bincount_spec(a, out):
    for i in range(len(a)):
        out[a[i]] += 1
        

# Consider a as a column vector tensor a row of j 1's
# Check where it is equal to a row [0, 1, 2, ...] tensor a column of 1's
# Then sum the columns
def bincount(a: TT["i"], j: int) -> TT["j"]:
    return ones(a.shape[0]) @ where(a[:,None] == arange(j)[None, :], 1, 0) 


def constraint_set_max(d):
    d["a"] = d["a"] % d["return"].shape[0]
    return d


#test_bincount = make_test("bincount",
#    bincount, bincount_spec, add_sizes=["j"], constraint=constraint_set_max
#)
#run_test(test_bincount)


# PUZZLE 16

def scatter_add_spec(values, link, out):
    for j in range(len(values)):
        out[link[j]] += values[j]
        
# First, generate a matrix with columns 0,1,2,3,4, and compare it with a matrix with key as rows, to get a "permutation" matrix which matches key/link to output
# Then, multiply it with values
def scatter_add(values: TT["i"], link: TT["i"], j: int) -> TT["j"]:
    return where(arange(j)[:,None] * ones(values.shape[0])[None, :] == link[None,:], 1,0) @ values




def constraint_set_max(d):
    d["link"] = d["link"] % d["return"].shape[0]
    return d


test_scatter_add = make_test("scatter_add",
    scatter_add, scatter_add_spec, add_sizes=["j"], constraint=constraint_set_max
)

#run_test(test_scatter_add)  ####### PRETTY SURE THIS WORKS BUT RUNNING INTO SOME BUG???



# PUZZLE 17

def flatten_spec(a, out):
    k = 0
    for i in range(len(a)):
        for j in range(len(a[0])):
            out[k] = a[i][j]
            k += 1

# Indexing tensor is [0, 0, 0, 1, 1, 1] and [1,2, 3, 1, 2, 3]
def flatten(a: TT["i", "j"], i:int, j:int) -> TT["i * j"]:
    return a[(arange(i*j) - arange(i*j) % j)//j, arange(i*j) % j]



#test_flatten = make_test("flatten", flatten, flatten_spec, add_sizes=["i", "j"])
#run_test(test_flatten)


# PUZZLE 18

def linspace_spec(i, j, out):
    for k in range(len(out)):
        out[k] = float(i + (j - i) * k / max(1, len(out) - 1))

# Have to deal with the annoying edge case where n = 1
def linspace(i: TT[1], j: TT[1], n: int) -> TT["n", float]:
    return arange(n) * (j-i) / where(n != 1, ones(1) * (n-1), ones(1)) + i

test_linspace = make_test("linspace", linspace, linspace_spec, add_sizes=["n"])
#run_test(test_linspace)


# PUZZLE 19

def heaviside_spec(a, b, out):
    for k in range(len(out)):
        if a[k] == 0:
            out[k] = b[k]
        else:
            out[k] = int(a[k] > 0)

def heaviside(a: TT["i"], b: TT["i"]) -> TT["i"]:
    return where(a == 0, b, 0) + where(a > 0, 1, 0)

test_heaviside = make_test("heaviside", heaviside, heaviside_spec)
#run_test(test_heaviside)


# PUZZLE 20

def repeat_spec(a, d, out):
    for i in range(d[0]):
        for k in range(len(a)):
            out[i][k] = a[k]

def constraint_set(d):
    d["d"][0] = d["return"].shape[0]
    return d

            
def repeat(a: TT["i"], d: TT[1]) -> TT["d", "i"]:
    return a[None, :][arange(d) * 0,:]

test_repeat = make_test("repeat", repeat, repeat_spec, constraint=constraint_set)
#run_test(test_repeat)


# PUZZLE 21

def bucketize_spec(v, boundaries, out):
    for i, val in enumerate(v):
        out[i] = 0
        for j in range(len(boundaries)-1):
            if val >= boundaries[j]:
                out[i] = j + 1
        if val >= boundaries[-1]:
            out[i] = len(boundaries)


def constraint_set(d):
    d["boundaries"] = np.abs(d["boundaries"]).cumsum()
    return d

# If the boundaries is [3, 6] then the buckets are (-inf, 3), [3, 6), [6, inf), labelled 0, 1, 2 respectively
def bucketize(v: TT["i"], boundaries: TT["j"]) -> TT["i"]:
    #return sum(where(boundaries[None,:] * ones(v.shape[0])[:,None] <= ones(boundaries.shape[0])[None,:] * v[:,None], 1, 0))
    return ones(boundaries.shape[0]) @ where(boundaries[:, None] * ones(v.shape[0])[None,:] <= ones(boundaries.shape[0])[:, None] * v[None, :], 1, 0)


test_bucketize = make_test("bucketize", bucketize, bucketize_spec,
                           constraint=constraint_set)
run_test(test_bucketize)









# SPEEDRUN (or really... "lengthrun")

import inspect
#fns = (ones, sum, outer, diag, eye, triu, cumsum, diff, vstack, roll, flip,
#       compress, pad_to, sequence_mask, bincount, scatter_add)
fns = (ones, sum, outer, diag, eye, triu, cumsum, diff, vstack, roll, flip,
       compress, pad_to, sequence_mask, bincount, scatter_add, flatten, linspace, heaviside, repeat, bucketize)



for fn in fns:
    lines = [l for l in inspect.getsource(fn).split("\n") if not l.strip().startswith("#")]
    
    if len(lines) > 3:
        print(fn.__name__, len(lines[2]), "(more than 1 line)")
    else:
        print(fn.__name__, len(lines[1]))


'''
ones 28
sum 39
outer 33
diag 52
eye 70
triu 70
cumsum 167
diff 117
vstack 94
roll 57
flip 31
compress 55
pad_to 39
sequence_mask 111
bincount 75
scatter_add 98
flatten 65
linspace 74
heaviside 51
repeat 38
bucketize 149
'''