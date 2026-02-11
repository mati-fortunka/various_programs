a=1
b=5

def sprawdz_liczby(x,y):
    if x==y:
        print(f"{x} równy {y}")
    elif x>y:
        print(f"{x} > {y}")
    else:
        print(f"{x} < {y}")

sprawdz_liczby(a,b)
sprawdz_liczby(b,b)
