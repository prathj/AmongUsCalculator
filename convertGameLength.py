
def convert(num):
    num = num[:2] + ':' + num[4:6]
    ftr = [60, 1]
    t = round(sum([a * b for a, b in zip(ftr, [int(i) for i in num.split(":")])]) / 60, 2)
    return t

