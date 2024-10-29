

def parseRM(text):
    words = text.split(' ')
    TC = 0
    RC = 0
    Vis = 0
    Hid = 0
    RecHid = False
    Seed = None
    P = None
    Q = None
    ETAV = None
    ETAH = None
    for w in words:
        if w[0:3] == 'TC:':
            TC = int(w[3:])

        elif w[0:3] == 'RC:':
            RC = int(w[3:])

        elif w[0:3] == 'TB:':
            TB = int(w[3:])

        elif w[0:4] == 'VIS:':
            Vis = int(w[4:])

        elif w[0:4] == 'HID:':
            Hid = int(w[4:])

        elif w[0:5] == 'Seed:':
            Seed = w

        elif w[0:7] == 'RecHid:':
            RecHid = bool(int(w[7]))

        elif w[0:2] == 'P:':
            P = float(w[2:])

        elif w[0:2] == 'Q:':
            Q = float(w[2:])

        elif w[0:5] == 'ETAV:':
            ETAV = float(w[5:])

        elif w[0:5] == 'ETAH:':
            ETAH = float(w[5:])

    return TC, TB, Vis, Hid, RC, Seed, RecHid, P, Q, ETAV, ETAH
