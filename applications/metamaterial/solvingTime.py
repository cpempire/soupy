import pickle


type_set = ['constant','linear','quadratic','saa']
size = 4
# meshsize_set = [1, 2, 3, 4, 5]
# mesh = ['dim = 940', 'dim = 3,336', 'dim = 12,487', 'dim = 48,288', 'dim = 189,736'] #["5,809", "20,097", "79,873", "31,8465", "1,271,809"]
meshsize_set = [0, 1, 2, 3, 4, 5]
mesh = ['dim = 312', 'dim = 995', 'dim = 3,469', 'dim = 12,904', 'dim = 49,667', 'dim = 194,765']

# [312, 995, 3469, 12904, 49667, 194765]
# [10466, 40058, 157426, 624746, 2488138, 9932874]

for meshsize in meshsize_set[0:size]:
    tobj = []
    tgrad = []
    trand = []
    for type in type_set:
        filename = "run_disk"+str(meshsize)+"/data/"+type+"/data_l1.p"
        data = pickle.load(open(filename, 'rb'))
        tobj.append(data["tobj"])
        tgrad.append(data["tgrad"])
        trand.append(data["trand"])

    print("tobj = ", tobj)
    print("tgrad = ", tgrad)
    print("trand = ", trand)