#!/usr/bin/env python2

#I have no idea where I'm going with this.
#And neither will you.

from __future__ import division
#this import is a guard against using 2.x division semantics instead of the
#saner 3.x division semantics. only int division is actually used.

import Image

#halve x, round to even for ties
def halve(x):
    #if type(x) not in (int,long):
    #    return x/2
    return (x + x%4//2)//2


def median(hist,total=None):
    '''Calculate the median of a histogram.'''
    if total is None:
        total = sum(hist)
    
    halftotal = (total+1)//2
    
    s = 0
    
    low = -1
    for i,v in enumerate(hist):
        if v > 0:
            s += v
            if s == halftotal:
                if total%2 == 1:
                    return i
                else:
                    low = i
            elif s > halftotal:
                if total%2 == 1 or low < 0:
                    return i
                else:
                    return halve(i+low)


def squaremedian(im,radius=1):
    '''Apply a square median filter with width = 2r+1 on a planar image. Border pixels are not processed. Uses the standard O(r) algorithm.'''
    
    width,height = im.size
    im_data = im.load()
    result = im.copy()
    result_data = result.load()
    
    window = 2*radius+1 # width of the window
    total = window**2 # total number of pixels in the window
    
    if window > width or window > height:
        #no need to process anything if all we have are borders
        return result
    
    for y in range(radius,height-radius):
        #initialise the histogram for every row we need to process
        hist = [0]*256
        for i in range(window):
            for j in range(-radius,radius+1):
                hist[im_data[i,y+j]] += 1
        result_data[radius,y] = median(hist,total)
        #then update and recompute the median for every pixel in the row
        for x in range(radius+1,width-radius):
            for i in range(-radius,radius+1):
                hist[im_data[x-radius-1,y+i]] -= 1
                hist[im_data[x+radius  ,y+i]] += 1
            result_data[x,y] = median(hist,total)
    
    return result

def median3(a,b,c):
    '''Median of three inputs. Average 8/3 comparisons for random input.'''
    if a > b:
        if b > c:
            return b
        elif a > c:
            return c
        else:
            return a
    else:
        if b < c:
            return b
        elif a < c:
            return c
        else:
            return a

def select(l,index,pivot=None):
    '''Non-inplace quickselect.'''
    if index == 0:
        return min(l)
    elif index == len(l)-1:
        return max(l)
    
    if pivot is None:
        pivot = median3(l[0],l[index],l[-1])
    
    lower  = []
    equal  = []
    higher = []
    
    for i,v in enumerate(l):
        if v < pivot:
            lower  += [v]
        elif v == pivot:
            equal  += [v]
        else:
            higher += [v]
    
    if len(lower) > index:
        return select(lower,index)
    index -= len(lower)
    if len(equal) > index:
        return pivot
    return select(higher,index-len(equal))
        
def listmedian(l,pivot=None):
    '''Return the median of a list.'''
    if len(l)%2 == 1:
        return select(l,len(l)//2,pivot)
    else:
        return halve(select(l,len(l)//2-1,pivot)+select(l,len(l)//2,pivot))

def squaremedian_select(im,radius=1):
    '''Apply a square median filter with width = 2r+1 on a planar image. Border pixels are not processed. Uses O(r^2) selection algorithm.'''
    
    width,height = im.size
    im_data = im.load()
    result = im.copy()
    result_data = result.load()
    
    window = 2*radius+1 # width of the window
    
    if window > width or window > height:
        return result
    
    for x in range(radius,width-radius):
        l = []
        for i in range(window):
            l += [im_data[X,i] for X in range(x-radius,x+radius+1)]
        result_data[x,radius] = p = listmedian(l)
        for y in range(radius+1,height-radius):
            l = l[window:]+[im_data[X,y+radius] for X in range(x-radius,x+radius+1)]
            result_data[x,y] = p = listmedian(l,p)
    
    return result

def addhist(h1,h2,n):
    return [h1[i]+h2[i] for i in range(n)]

def subhist(h1,h2,n):
    return [h1[i]-h2[i] for i in range(n)]

def squaremedian_ctmf(im,radius=1):
    '''Apply a square median filter with width = 2r+1 on a planar image. Border pixels are not processed. Uses a modified O(1) algorithm from http://nomis80.org/ctmf.pdf .'''
    
    width,height = im.size
    im_data = im.load()
    result = im.copy()
    result_data = result.load()
    
    window = 2*radius+1 # width of the window
    total = window**2
    
    if window > width or window > height:
        return result
    
    h = []
    for x in range(width):
        h += [[0]*256]
        for y in range(window-1):
            h[x][im_data[x,y]] += 1
    
    for y in range(radius,height-radius):
        for x in range(width):
            h[x][im_data[x,y+radius]] += 1
        H = [0]*256
        for x in range(window-1):
            for i in range(256):
                H[i] += h[x][i]
        for x in range(radius,width-radius):
            H = addhist(H,h[x+radius],256)
            result_data[x,y] = median(H,total)
            H = subhist(H,h[x-radius],256)
        for x in range(width):
            h[x][im_data[x,y-radius]] -= 1
    #rather than messily handling which variables to update and which not to
    #around loops, we can use half-updates at the start and end of the loops
    #instead. using serpentine scanning, this can be made slightly faster, but
    #at the cost of code simplicity.
    
    return result


def binomialcoefficients(n):
    if n < 0:
        return [1]
    c = [1]
    for i in range(1,n+1):
        c += [c[-1]*(n-i+1)//i]
    return c

def binmedian(im,radius=1):
    '''Apply a binomial-weighted median filter with width = 2r+1 on a planar image. Border pixels are not processed.'''
    
    width,height = im.size
    im_data = im.load()
    result = im.copy()
    result_data = result.load()
    
    window = 2*radius+1
    total = 16**radius #(2**(2*radius))**2
    kernel = binomialcoefficients(2*radius)
    
    if window > width or window > height:
        return result
    
    for y in range(radius,height-radius):
        for x in range(radius,width-radius):
            #unlike with the equally-weighted square filter, we can't optimise this as easily
            #the binomial structure seems to suggest an O(r) implementation is possible,
            #but that's not trivial.
            hist = [0]*256
            for i in range(-radius,radius+1):
                for j in range(-radius,radius+1):
                    hist[im_data[x+i,y+j]] += kernel[i+radius]*kernel[j+radius]
            result_data[x,y] = median(hist,total)
    
    return result

#for multi-channel images, which seem to be stored in an interleaved format
def applyplanarfilter(im,f):
    return Image.merge(im.mode,[f(plane) for plane in im.split()])

if __name__ == '__main__':
    #some ad hoc testing because hurf
    im = Image.open('../test-random-mono.bmp')
    squaremedian_ctmf(im,radius=50).save('test-random-mono-squaremedian101-ctmf.png')
    squaremedian     (im,radius=50).save('test-random-mono-squaremedian101-std.png')


''' some notes:
selection is fastest for r<3, standard algorithm is fastest until r<60 or so,
and thereafter the constant-time algorithm is fastest. which is a bit surprising
because the paper introducing the constant-time algorithm actually had it much
faster than the standard algorithm even for relatively small radii.

this might or might not have something to do with the fact that I failed to get
this working with PyPy. I blame PIL.
'''
