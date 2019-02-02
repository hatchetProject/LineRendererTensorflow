

def BresenhamLine(x0, y0, x1, y1):
    dx = x1 - x0
    dy = y1 - y0
    if dx > 0:
        ux = 1
    else:
        ux = -1

    if dy > 0:
        uy = 1
    else:
        uy = -1

    dx2 = dx / 2
    dy2 = dy / 2
    res = []
    if (abs(dx) > abs(dy)):
        e = -dx
        x = x0
        y = y0
        for x in range (x0, x1, ux):
            res.append(x)
            res.append(y)
            e = e + dy2
            if e > 0:
                y += uy
                e = e - dx2
    elif (abs(dx) <= abs(dy) and abs(dx) != 0):
        e = -dy
        x = x0
        y = y0
        for y in range(y0, y1, uy):
            res.append(x)
            res.append(y)
            e = e + dx2
            if e > 0:
                x += ux
                e = e - dy2
    else:
        if y0 < y1:
            uy = 1
        else:
            uy = -1
        for y in range(y0, y1, uy):
            res.append(x0)
            res.append(y)
    return res