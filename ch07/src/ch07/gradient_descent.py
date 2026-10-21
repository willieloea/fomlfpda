def dot(w, d):
    return sum(w_i*d_i for w_i, d_i in zip(w,d))



if __name__ == '__main__':
    x = [1,2,3]
    y = [1,2,3]
    print(dot(x,y))
