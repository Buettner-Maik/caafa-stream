import os, sys, argparse
sys.path.append(os.path.join(os.getcwd(), '.'))

import additions.constants as consts
import numpy as np
import pandas as pd

def gen_mixed_dataset():
    folder = os.path.join(consts.DIR_CSV, 'corr', 'mixed')
    filename = os.path.join(folder, 'raw_data.pkl.gzip')
    if not os.path.exists(folder):
        os.mkdir(folder)
        
    # 8 features, 4 num, 4 cat (n1, n2, n3, n4, c5, c6, c7, c8)
    # n0 € [0..1] Useless feature
    # n1 € [0..1]
    # n2 € 2 * n1 ± N(0, 0.2)
    # n3 € [0..1]
    # n4 € f(c8) -> [a: 0.1, b: 0.3, c: 0.5, d: 0.7, e: 0.9] ± N(0, 0.2)
    # c5 € [a, b, c, d, e]
    # c6 € upper(c5) with 20% chance of random change to another category
    # c7 € f(n3) -> {a:0..<0.2, b:0.2..<0.4, c:0.4..<0.6, d:0.6..<0.8, e:0.8..1.0} with 20% chance of random change to another category
    # c8 € [a, b, c, d, e]
    # c9 € [a, b, c, d, e] Useless feature
    
    d = 1000
    r = 0.2
    
    n0 = np.random.random((d, 1))
    n1 = np.random.random((d, 1))
    n2 = n1 * 2 + np.random.normal(0, r, (d, 1))
    n3 = np.random.random((d, 1))
    
    def ccol(ra=None):
        c = np.array([['']*d]).T
        if ra is None: ra = np.random.random((d, 1))
        np.putmask(c, ra < 0.2, 'a')
        np.putmask(c, (ra >= 0.2) & (ra < 0.4), 'b')
        np.putmask(c, (ra >= 0.4) & (ra < 0.6), 'c')
        np.putmask(c, (ra >= 0.6) & (ra < 0.8), 'd')
        np.putmask(c, ra > 0.8, 'e')
        return c

    c5 = ccol()
    c6 = c5.copy()
    np.putmask(c6, np.random.random((d, 1)) < 0.2, ccol())
    c6[:, 0] = np.apply_along_axis(lambda x: x[0].upper(), 1, c6)
    c7 = ccol(ra=n3)
    np.putmask(c7, np.random.random((d, 1)) < 0.2, ccol())
    c8 = ccol()
    c9 = ccol()

    n4 = np.zeros((d, 1))
    np.putmask(n4, c8 == 'a', .1)
    np.putmask(n4, c8 == 'b', .3)
    np.putmask(n4, c8 == 'c', .5)
    np.putmask(n4, c8 == 'd', .7)
    np.putmask(n4, c8 == 'e', .9)
    n4 += np.random.normal(0, r, (d, 1))

    print(n0.shape, n1.shape, n2.shape, n3.shape, n4.shape)
    print(c5.shape, c6.shape, c7.shape, c8.shape, c9.shape)

    data = np.concatenate([n0, n1, n2, n3, n4, c5, c6, c7, c8, c9], axis=1)
    cols = ['N0', 'N1', 'N2', 'N3', 'N4', 'C5', 'C6', 'C7', 'C8', 'C9']
    
    df = pd.DataFrame(data, columns=cols)
    df[cols[:5]] = df[cols[:5]].astype(float)
    df.to_pickle(filename)

    print(df)

def _lin_incr(begin, end, i_begin, i_end, chunk):
    vals = i_end - i_begin + 1
    srange = end - begin
    step = srange / vals#(vals - 1)
    #moving_values = np.arange(begin, end * (1 + 1 / (vals + 1)), step)
    moving_values = np.arange(begin, end, step)
    return np.append(np.append([begin]*i_begin, moving_values), [end]*(chunk - i_end - 1))
def _gradual(concept1, concept2, i_begin, i_end, chunk):
    if not isinstance(concept1, np.ndarray):
        if isinstance(concept1, pd.Series): concept1 = concept1.to_numpy()
        else: concept1 = np.array([concept1]*chunk)
    if not isinstance(concept2, np.ndarray): 
        if isinstance(concept1, pd.Series): concept2 = concept2.to_numpy()
        else: concept2 = np.array([concept2]*chunk)
    chances = _lin_incr(0, 1, i_begin, i_end, chunk)
    mask = np.random.random(chunk) < chances
    t = concept1.copy()
    np.putmask(t, mask, concept2)
    return t

def change_label_func(filepath, class_col='class'):
    df = pd.read_pickle(filepath)
    
    def re_class(b:int, e:int, ccol:str, true_ncols:list, false_ncols:list, 
                 true_coefs:list, false_coefs:list, thrs:list, revs:list, 
                 flip_chance:float=.0, df:pd.DataFrame=df, lcol:str=class_col):
        t = np.zeros(e - b + 1).astype('bool')
        temp = np.zeros(e - b + 1)

        for ncol, coef in zip(true_ncols, true_coefs):
            temp += df.loc[b:e, ncol] * coef
        temp = (temp < thrs[0]) ^ revs[0]
        if isinstance(ccol, str): temp = (df.loc[b:e, ccol] == 'b') & temp
        else: temp = (ccol == 'b') & temp
        t = t | temp

        for ncol, coef in zip(false_ncols, false_coefs):
            temp += df.loc[b:e, ncol] * coef
        temp = (temp < thrs[1]) ^ revs[1]
        if isinstance(ccol, str): temp = (df.loc[b:e, ccol] != 'b') & temp
        else: temp = (ccol != 'b') & temp
        t = t | temp
        
        flip = np.random.random(e - b + 1) < flip_chance
        print(flip_chance, flip.sum())
        df.loc[b:e, lcol] = t ^ flip
    
    re_class(0, 9999, 'c7', ['n2', 'n3'], ['n3', 'n4'], 
        [1, 1], [1, 1], [.5, -.5], 
        [False, False], 1 / 32)
    re_class(10000, 10999, 'c7', ['n2', 'n3'], ['n3', 'n4'],
        [1, 1], [1, 1], [_lin_incr(.5, -1, 0, 499, 1000), _lin_incr(-.5, 0, 0, 499, 1000)],
        [False, False], 1 / 32)
    grad = _gradual(df.loc[11000:11999, 'c7'], df.loc[11000:11999, 'c6'], 0, 499, 1000)
    grad = pd.Series(grad, index=range(11000, 12000))
    re_class(11000, 11999, grad, ['n2', 'n3'], ['n3', 'n4'], 
        [1, 1], [1, 1], [-1, 0], 
        [False, False], 1 / 32)
    re_class(12000, 12999, 'c5', ['n0', 'n4'], ['n1'], 
        [1, 1], [1, 1], [.5, .5], 
        [False, True], 1 / 32)

    df[class_col] = df[class_col].replace({0:'A', 1:'B'})

    return df

def change_label_func_splits(filepath, class_col='class'):
    df = pd.read_pickle(filepath)
    df[class_col] = df[class_col].replace({'A':0, 'B':1})
    folder = os.path.join(os.path.dirname(filepath), 'prepared')

    for i in range(30):
        for m in [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]:
            datapath = os.path.join(folder, str(i), str(m), 'labeled_set.pkl.gzip')
            data = pd.read_pickle(datapath)
            data.loc[data.index, class_col] = df.loc[data.index, class_col]
            data.to_pickle(datapath)
            for num in range(0, 12950, 50):
                datapath = os.path.join(folder, str(i), str(m), f'data{num}.pkl.gzip')
                data = pd.read_pickle(datapath)
                data.loc[data.index, class_col] = df.loc[data.index, class_col]
                data.to_pickle(datapath)



def cfpdss_dataset():
    folder = os.path.join(consts.DIR_CSV, 'cfpdss')
    filename = os.path.join(folder, 'raw_data.pkl.gzip')
    # 10 features, 1 label 2 classes
    # n0 € N1
    # n1 € N1
    # n2 € a * (n1 ± N.25)
    # n3 € N1
    # n4 € a * n1 + (1 - a) * n3
    # c5 € [a, b, c, d], no correlation
    # c6 € [a, b, c, d]
    # c7 € c5 with 25% chance to reroll random category
    # c8 € f(n3) -> {a:0..<0.25, b:0.25..<0.5, c:0.5..<0.75, d:0.75..<1.0} with 25% chance to reroll random category
    # c9 € [a, b, c, d]
    # l € {0, 1} c7 n2

    dataset = np.zeros((0, 11))
    chunk = 1000
    t_end = chunk // 2 - 1
    err = 0.125

    def lin_incr(begin, end, i_begin=0, i_end=t_end):
        return _lin_incr(begin, end, i_begin, i_end, chunk)
    def gradual(concept1, concept2, i_begin=0, i_end=t_end):
        return _gradual(concept1, concept2, i_begin, i_end, chunk)
    def tfunc(ccol1, ccol2, ncols1, ncols2, ncols3, ncols4, 
        coefs1, coefs2, coefs3, coefs4, thrs, revs, flip_chance=0):
        t = np.zeros(chunk).astype('bool')

        temp = np.zeros(chunk)
        for ncol, coef in zip(ncols1, coefs1):
            temp += ncol * coef
        temp = ccol1 & ccol2 & ((temp < thrs[0]) ^ revs[0])
        t = t | temp

        temp = np.zeros(chunk)
        for ncol, coef in zip(ncols2, coefs2):
            temp += ncol * coef
        temp = ccol1 & (~ccol2) & ((temp < thrs[1]) ^ revs[1])
        t = t | temp

        temp = np.zeros(chunk)
        for ncol, coef in zip(ncols3, coefs3):
            temp += ncol * coef
        temp = (~ccol1) & ccol2 & ((temp < thrs[2]) ^ revs[2])
        t = t | temp

        temp = np.zeros(chunk)
        for ncol, coef in zip(ncols4, coefs4):
            temp += ncol * coef
        temp = (~ccol1) & (~ccol2) & ((temp < thrs[3]) ^ revs[3])
        t = t | temp

        flip = np.random.random(chunk) < flip_chance
        return np.logical_xor(t, flip)

    def ncol(stddev=1, ra=None, ra2=None, mult_ra=1, mult_ra2=1):
        if stddev == 0: noise = np.zeros(chunk)
        else: noise = np.random.normal(scale=stddev, size=chunk)
        #else: noise = np.random.uniform(high=stddev, size=chunk)
        if ra is None: return noise
        elif ra2 is None: return mult_ra * (ra + noise)
        else: return (mult_ra * ra) + (mult_ra2 * ra2) + noise
    def ccol(flip_chance=0, ra=None, ra2=None, ra_thr=0, reverse_cats=False):
        if ra is None: 
            ra = np.random.random(chunk)
            c = ra < ra_thr
        elif ra2 is None:
            if np.issubdtype(ra.dtype, np.number): c = ra < ra_thr
            else: c = ra
        else: c = np.logical_xor(ra, ra2)
        
        flip = np.random.random(chunk) < flip_chance
        c = np.logical_xor(c, flip)
        if reverse_cats: c = ~c
        return c

    # n0 = ncol()
    # n1 = ncol()
    # n2 = ncol(0, n1, mult_ra=2)
    # n3 = ncol()
    # n4 = ncol(0, n1, n3, .5, .5)
    # c5 = ccol(ra_thr=.5)
    # c6 = ccol(ra_thr=.5)
    # c7 = ccol(0, c6)
    # c8 = ccol(0, n3, ra_thr=0)
    # c9 = ccol(0, c6, c8)
    # t = tfunc(c7, c7, [n2], [n2], [n2], [n2], [1], [1], [1], [1], 
    #     [.5, 0, 0, .5], [False, False, False, True]) #c7:n2<.5, ~c7:n2>.5
    # tenbatch = np.concatenate([n0, n1, n2, n3, n4, c5, c6, c7, c8, c9, t], axis=1)
    # dataset = np.concatenate([dataset, tenbatch])
    # pd.DataFrame(dataset).to_csv('bla.csv', sep='\t')
    # print(dataset)
    # #print('0',n0,'1',n1,'2',n2,'3',n3,'4',n4,'5',c5,'6',c6,'7',c7,'8',c8,'9',c9,'C',t)
    # raise

    # 0..999
    n0 = ncol()
    n1 = ncol()
    n2 = ncol(err, n1, mult_ra=2)
    n3 = ncol()
    n4 = ncol(err, n1, n3, .5, .5)
    c5 = ccol(ra_thr=.5)
    c6 = ccol(ra_thr=.5)
    c7 = ccol(err, c6)
    c8 = ccol(err, n3, ra_thr=0)
    c9 = ccol(err, c6, c8)
    t = tfunc(c7, c7, [n2, n3], [n2, n3], [n2, n3], [n2, n3], [1, 1], [1, 1], [1, 1], [1, 1], 
        [.5, 0, 0, .5], [False, False, False, True], err) #c7:n2<.5, ~c7:n2>.5
    tenbatch = np.array([n0, n1, n2, n3, n4, c5, c6, c7, c8, c9, t]).T
    dataset = np.concatenate([dataset, tenbatch])

    # 1000..1999 inc num f
    n0 = ncol() + lin_incr(0, 2) #to 2
    n1 = ncol() + lin_incr(0, 1) #to 1
    n2 = ncol(err, n1, mult_ra=2)
    n3 = ncol() + lin_incr(0, -1) #to -1
    n4 = ncol(err, n1, n3, .5, .5)
    c5 = ccol(ra_thr=.5)
    c6 = ccol(ra_thr=.5)
    c7 = ccol(err, c6)
    c8 = ccol(err, n3, ra_thr=0)
    c9 = ccol(err, c6, c8)
    t = tfunc(c7, c7, [n2, n3], [n2, n3], [n2, n3], [n2, n3], [1, 1], [1, 1], [1, 1], [1, 1], 
        [.5, 0, 0, .5], [False, False, False, True], err) #c7:n2<.5, ~c7:n2>.5
    tenbatch = np.array([n0, n1, n2, n3, n4, c5, c6, c7, c8, c9, t]).T
    dataset = np.concatenate([dataset, tenbatch])

    # 2000..2999 inc cat f
    n0 = ncol() + 2 
    n1 = ncol() + 1
    n2 = ncol(err, n1, mult_ra=2)
    n3 = ncol() - 1
    n4 = ncol(err, n1, n3, .5, .5)
    c5 = ccol(ra_thr=lin_incr(.5, .8)) #to .8
    c6 = ccol(ra_thr=lin_incr(.5, .2)) #to .2
    c7 = ccol(err, c6)
    c8 = ccol(err, n3, ra_thr=lin_incr(0, -1)) #to -1
    c9 = ccol(err, c6, c8)
    t = tfunc(c7, c7, [n2, n3], [n2, n3], [n2, n3], [n2, n3], [1, 1], [1, 1], [1, 1], [1, 1], 
        [.5, 0, 0, .5], [False, False, False, True], err) #c7:n2<.5, ~c7:n2>.5
    tenbatch = np.array([n0, n1, n2, n3, n4, c5, c6, c7, c8, c9, t]).T
    dataset = np.concatenate([dataset, tenbatch])

    # 3000..3999 inc coef f
    n0 = ncol() + 2 
    n1 = ncol() + 1
    n2 = ncol(err, n1, mult_ra=lin_incr(2, 5)) #to 5
    n3 = ncol() - 1
    n4 = ncol(err, n1, n3, lin_incr(.5, .8), lin_incr(.5, .2)) #to (.8, .2)
    c5 = ccol(ra_thr=.8)
    c6 = ccol(ra_thr=.2)
    c7 = ccol(lin_incr(err, .25), c6) #to .25
    c8 = ccol(err, n3, ra_thr=-1)
    c9 = ccol(lin_incr(err, .25), c6, c8) #to .25
    t = tfunc(c7, c7, [n2, n3], [n2, n3], [n2, n3], [n2, n3], [1, 1], [1, 1], [1, 1], [1, 1], 
        [.5, 0, 0, .5], [False, False, False, True], err) #c7:n2<.5, ~c7:n2>.5
    tenbatch = np.array([n0, n1, n2, n3, n4, c5, c6, c7, c8, c9, t]).T
    dataset = np.concatenate([dataset, tenbatch])

    # 4000..4999 gra num f
    n0 = ncol() + gradual(2, -2) #to -2
    n1 = ncol() + gradual(1, -1) #to -1
    n2 = ncol(err, n1, mult_ra=5)
    n3 = ncol() + gradual(-1, 1) #to 1
    n4 = ncol(err, n1, n3, .8, .2)
    c5 = ccol(ra_thr=.8)
    c6 = ccol(ra_thr=.2)
    c7 = ccol(.25, c6)
    c8 = ccol(err, n3, ra_thr=-1)
    c9 = ccol(.25, c6, c8)
    t = tfunc(c7, c7, [n2, n3], [n2, n3], [n2, n3], [n2, n3], [1, 1], [1, 1], [1, 1], [1, 1], 
        [.5, 0, 0, .5], [False, False, False, True], err) #c7:n2<.5, ~c7:n2>.5
    tenbatch = np.array([n0, n1, n2, n3, n4, c5, c6, c7, c8, c9, t]).T
    dataset = np.concatenate([dataset, tenbatch])

    # 5000..5999 gra cat f
    n0 = ncol() - 2
    n1 = ncol() - 1
    n2 = ncol(err, n1, mult_ra=5)
    n3 = ncol() + 1
    n4 = ncol(err, n1, n3, .8, .2)
    c5 = ccol(ra_thr=gradual(.8, .2)) #to .2
    c6 = ccol(ra_thr=gradual(.2, .8)) #to .8
    c7 = ccol(.25, c6)
    c8 = ccol(err, n3, ra_thr=gradual(-1, 1)) #to 1
    c9 = ccol(.25, c6, c8)
    t = tfunc(c7, c7, [n2, n3], [n2, n3], [n2, n3], [n2, n3], [1, 1], [1, 1], [1, 1], [1, 1], 
        [.5, 0, 0, .5], [False, False, False, True], err) #c7:n2<.5, ~c7:n2>.5
    tenbatch = np.array([n0, n1, n2, n3, n4, c5, c6, c7, c8, c9, t]).T
    dataset = np.concatenate([dataset, tenbatch])

    # 6000..6999 gra coef f
    n0 = ncol() - 2
    n1 = ncol() - 1
    n2 = ncol(err, n1, mult_ra=gradual(5, -3)) #to -3
    n3 = ncol() + 1
    n4 = ncol(err, n1, n3, gradual(.8, .2), gradual(.2, .8)) #to (.2, .8)
    c5 = ccol(ra_thr=.2)
    c6 = ccol(ra_thr=.8)
    c7 = ccol(gradual(.25, .375), c6) #to .375
    c8 = ccol(err, n3, ra_thr=1)
    c9 = ccol(.25, gradual(c6, c5), c8) #to c5
    t = tfunc(c7, c7, [n2, n3], [n2, n3], [n2, n3], [n2, n3], [1, 1], [1, 1], [1, 1], [1, 1], 
        [.5, 0, 0, .5], [False, False, False, True], err) #c7:n2<.5, ~c7:n2>.5
    tenbatch = np.array([n0, n1, n2, n3, n4, c5, c6, c7, c8, c9, t]).T
    dataset = np.concatenate([dataset, tenbatch])

    # 7000..7999 sudden num f
    n0 = ncol() #to 0
    n1 = ncol() #to 0
    n2 = ncol(err, n1, mult_ra=-3)
    n3 = ncol() #to 0
    n4 = ncol(err, n1, n3, .2, .8)
    c5 = ccol(ra_thr=.2)
    c6 = ccol(ra_thr=.8)
    c7 = ccol(.375, c6)
    c8 = ccol(err, n3, ra_thr=1)
    c9 = ccol(.25, c5, c8)
    t = tfunc(c7, c7, [n2, n3], [n2, n3], [n2, n3], [n2, n3], [1, 1], [1, 1], [1, 1], [1, 1], 
        [.5, 0, 0, .5], [False, False, False, True], err) #c7:n2<.5, ~c7:n2>.5
    tenbatch = np.array([n0, n1, n2, n3, n4, c5, c6, c7, c8, c9, t]).T
    dataset = np.concatenate([dataset, tenbatch])
    
    # 8000..8999 sudden num f
    n0 = ncol()
    n1 = ncol()
    n2 = ncol(err, n1, mult_ra=-3)
    n3 = ncol()
    n4 = ncol(err, n1, n3, .2, .8)
    c5 = ccol(ra_thr=.5) #to .5
    c6 = ccol(ra_thr=.5) #to .5
    c7 = ccol(.375, c6)
    c8 = ccol(err, n3, ra_thr=0) #to 0
    c9 = ccol(.25, c5, c8)
    t = tfunc(c7, c7, [n2, n3], [n2, n3], [n2, n3], [n2, n3], [1, 1], [1, 1], [1, 1], [1, 1], 
        [.5, 0, 0, .5], [False, False, False, True], err) #c7:n2<.5, ~c7:n2>.5
    tenbatch = np.array([n0, n1, n2, n3, n4, c5, c6, c7, c8, c9, t]).T
    dataset = np.concatenate([dataset, tenbatch])

    # 9000..9999 sudden coef f
    n0 = ncol()
    n1 = ncol()
    n2 = ncol(err, n1, mult_ra=2) #to 2
    n3 = ncol()
    n4 = ncol(err, n1, n3, .5, .5) #to (.5, .5)
    c5 = ccol(ra_thr=.5) 
    c6 = ccol(ra_thr=.5)
    c7 = ccol(err, c6) #to .125
    c8 = ccol(err, n3, ra_thr=0)
    c9 = ccol(err, c6, c8) #to .125, c6
    t = tfunc(c7, c7, [n2, n3], [n2, n3], [n2, n3], [n2, n3], [1, 1], [1, 1], [1, 1], [1, 1], 
        [.5, 0, 0, .5], [False, False, False, True], err) #c7:n2<.5, ~c7:n2>.5
    tenbatch = np.array([n0, n1, n2, n3, n4, c5, c6, c7, c8, c9, t]).T
    dataset = np.concatenate([dataset, tenbatch])

    # 10000..10999 incremental label
    n0 = ncol()
    n1 = ncol()
    n2 = ncol(err, n1, mult_ra=2)
    n3 = ncol()
    n4 = ncol(err, n1, n3, .5, .5)
    c5 = ccol(ra_thr=.5) 
    c6 = ccol(ra_thr=.5)
    c7 = ccol(err, c6)
    c8 = ccol(err, n3, ra_thr=0)
    c9 = ccol(err, c6, c8)
    t = tfunc(c7, c7, [n2, n3], [n2, n3], [n2, n3], [n2, n3], [1, 1], [1, 1], [1, 1], [1, 1], 
        [lin_incr(.5, -.5), 0, 0, lin_incr(.5, -.5)], [False, False, False, True], err) #c7:n2<.5, ~c7:n2>.5
    tenbatch = np.array([n0, n1, n2, n3, n4, c5, c6, c7, c8, c9, t]).T
    dataset = np.concatenate([dataset, tenbatch])

    # 11000..11999 gradual label
    n0 = ncol()
    n1 = ncol()
    n2 = ncol(err, n1, mult_ra=2)
    n3 = ncol()
    n4 = ncol(err, n1, n3, .5, .5)
    c5 = ccol(ra_thr=.5) 
    c6 = ccol(ra_thr=.5)
    c7 = ccol(err, c6)
    c8 = ccol(err, n3, ra_thr=0)
    c9 = ccol(err, c6, c8)
    grad = gradual(c7, c6)
    t = tfunc(grad, grad, [n2, n3], [n2, n3], [n2, n3], [n2, n3], [1, 1], [1, 1], [1, 1], [1, 1], 
        [-.5, 0, 0, -.5], [False, False, False, True], err) #c7:n2<.5, ~c7:n2>.5
    tenbatch = np.array([n0, n1, n2, n3, n4, c5, c6, c7, c8, c9, t]).T
    dataset = np.concatenate([dataset, tenbatch])

    # 12000..12999 sudden label
    n0 = ncol()
    n1 = ncol()
    n2 = ncol(err, n1, mult_ra=2)
    n3 = ncol()
    n4 = ncol(err, n1, n3, .5, .5)
    c5 = ccol(ra_thr=.5) 
    c6 = ccol(ra_thr=.5)
    c7 = ccol(err, c6)
    c8 = ccol(err, n3, ra_thr=0)
    c9 = ccol(err, c6, c8)
    t = tfunc(c5, c5, [n0, n4], [n0, n4], [n0, n4], [n0, n4], [2, 1], [2, 1], [2, 1], [2, 1], 
        [1, 0, 0, 1], [False, False, False, True], err) #c7:n2<.5, ~c7:n2>.5
    tenbatch = np.array([n0, n1, n2, n3, n4, c5, c6, c7, c8, c9, t]).T
    dataset = np.concatenate([dataset, tenbatch])

    num_cols = ['n0', 'n1', 'n2', 'n3', 'n4']
    cat_cols = ['c5', 'c6', 'c7', 'c8', 'c9']
    label = ['class']
    df_dataset = pd.DataFrame(dataset, columns=num_cols + cat_cols + label)
    df_dataset[cat_cols] = df_dataset[cat_cols].replace({0:'a', 1:'b'})
    df_dataset[label] = df_dataset[label].replace({0:'A', 1:'B'})
    df_dataset.to_csv(f'{filename}.csv', index=False)
    df_dataset.to_pickle(filename)

def main():

    error_dists = ['normal', 'uniform']
    corr_funcs = ['sum', 'mean', 'min', 'max']

    parser = argparse.ArgumentParser(description='Generates a dataset of a correlated feature and noise')
    parser.add_argument('-d', required=True, type=int, help='The no. of instances of the data set')
    parser.add_argument('-f', required=True, type=int, help='The no. of noise features')
    parser.add_argument('-c', required=True, type=int, help='The no. of correlating features')
    parser.add_argument('-r', required=True, type=float, help='The ± range of the random component')
    parser.add_argument('--error', required=False, choices=error_dists, default=error_dists[0], help='The type of error')
    parser.add_argument('--cfunc', required=False, choices=corr_funcs, default=corr_funcs[0], help='The function that fuses the correlated features')
    parser.add_argument('--coffset', required=False, type=float, default=0.0, help='A constant offset for correlated features')
    parser.add_argument('--ccoeff', required=False, type=float, default=2.0, help='A factor for correlated features')
    parser.add_argument('--cexpo', required=False, type=float, default=1.0, help='An exponential for correlated features')
    parser.add_argument('--cquant', required=False, type=int, default=0, help='If set will quantize the correlated feature by multiplying, flooring and dividing with this value')
    parser.add_argument('--filepath', required=False, default=None, help='Specify to set a custom save name')
    args = parser.parse_args()

    if args.f < 0:
        raise ValueError(f'Negative feature values are prohibited (f = {args.f})')
    if args.c < 0:
        raise ValueError(f'Negative feature values are prohibited (c = {args.c})')
    if args.c + args.f == 0:
        raise ValueError('At least one feature has to be generated')

    folder = os.path.join(consts.DIR_CSV, 'corr', f'f{args.f}_c{args.c}_r{args.r}') if args.filepath is None else os.path.dirname(args.filepath)
    filename = os.path.join(folder, 'raw_data.pkl.gzip') if args.filepath is None else args.filepath
    if not os.path.exists(folder):
        os.mkdir(folder)

    noise = np.random.random([args.d, args.f])
    corrs = np.random.random([args.d, args.c])

    if args.cfunc == corr_funcs[0]:
        fcorr = corrs.sum(axis=1, keepdims=True)
    elif args.cfunc == corr_funcs[1]:
        fcorr = corrs.mean(axis=1, keepdims=True)
    elif args.cfunc == corr_funcs[2]:
        fcorr = corrs.min(axis=1, keepdims=True)
    elif args.cfunc == corr_funcs[3]:
        fcorr = corrs.max(axis=1, keepdims=True)
    
    c_true = fcorr ** args.cexpo * args.ccoeff + args.coffset

    if args.error == error_dists[0]:
        c = c_true + np.random.normal(0, args.r, [args.d, 1])
    elif args.error == error_dists[1]:
        c = c_true + np.random.uniform(-args.r, args.r, [args.d, 1])
    
    if args.cquant != 0:
        c = np.floor(c * args.cquant) / args.cquant
    
    label = (c > c_true).reshape([args.d, 1])

    cols = [f'N{x}' for x in range(args.f)] + [f'C{x}' for x in range(args.c)] + ['T', 'L']
    data = np.concatenate((noise, corrs, c, label), axis=1)

    df = pd.DataFrame(data, columns=cols)
    df.to_pickle(filename)

if __name__ == '__main__':
    filepath = os.path.join(consts.DIR_CSV, "cfpdss//raw_data.pkl.gzip")
    change_label_func_splits(filepath)
    #df = change_label_func(filepath)
    #cfpdss_dataset()
    #gen_mixed_dataset()
    #main()