import numpy as np
import math 
import pygame as pg 
from perlin_noise import PerlinNoise 

pg.init()
width = 800
height = 500

screen = pg.display.set_mode((width, height))
pg.display.set_caption('Title')
clock = pg.time.Clock() 
noise = PerlinNoise(octaves=4 , seed = 10)

running  = True

def arrow(x,y,l,theta , col) : 
    # if l < 0 : theta = -theta
    aroTheta = .4
    arrowArm = 5
    # col = col,0,0
    x1,y1 = x,y
    x12 = x1 + l*math.cos(theta)
    y12 = y1 + l*math.sin(theta)
    l=l+arrowArm-1
    x2 = x1 + (l)*math.cos(theta)
    y2 = y1 + (l)*math.sin(theta)
    arrow1X = x2 - (arrowArm)*math.cos(theta + aroTheta)
    arrow1Y = y2 - (arrowArm)*math.sin(theta + aroTheta)
    arrow2X = x2 - (arrowArm)*math.cos(theta - aroTheta)
    arrow2Y = y2 - (arrowArm)*math.sin(theta - aroTheta)
    pg.draw.line(screen , (col) , (x1,y1) , (x12,y12) , 1)
    pg.draw.polygon(screen , (col) , [(x2,y2) , (arrow1X , arrow1Y) , (arrow2X , arrow2Y)] )
def dotArrow(x , y , l, theta , col) :
    x1,y1 = x,y
    x2,y2 = x + l*math.cos(theta) , y + l*math.sin(theta)
    col = col ,0,0
    pg.draw.line(screen , (col) , (x1,y1) , (x2,y2) , 1)
    pg.draw.circle(screen , (col) , (x2,y2), 2)

def scale(val , startX , endX , startY , endY) : 
    x1 , y1 = startX , startY
    x2 , y2 = endX , endY
    return y2 - (y2-y1)*(x2-val)/(x2-x1)
def color(value , range) : 
    hue = min(255, scale(value , 0 , range , 0,255))
    r = hue
    b = 255 - hue
    g = 255 - abs(r-b)
    return [r , g , b]




    
theta = np.pi/4
gap = 20
numCols = width//gap
numRows = height//gap
numCells = numCols * numRows
dyeGap = gap
dyeCols = width // dyeGap
dyeRows = height // dyeGap
dyeCells = dyeCols * dyeRows
dye = np.zeros(dyeCells)
u = np.zeros(numCells)
v = np.zeros(numCells)
s = np.ones(numCells)
for j in range(numRows) : 
    for i in range(numCols) : 
        firstCol = j*numCols + 0 
        lastCol = j*numCols + numCols -1
        firstRow = 0*numCols + i 
        lastRow = (numRows-1)*numCols + i 
        s[firstCol] = 0
        s[lastCol] = 0
        s[firstRow] = 0
        s[lastRow] = 0
def drawS():
    for j in range(numRows)  :
        for i in range(numCols) : 
            x,y = i*gap , j *gap
            col = s[j*numCols + i] * 255
            col = col,col,col
            pg.draw.rect(screen , col , pg.Rect(x,y,gap,gap),0)
def drawVels(): 
    for j in range(numRows) : 
        for i in range(numCols) :
            x , y = i*gap + gap//2 , j*gap + gap//2
            vecx = u[j*numCols + i]
            vecy = v[j*numCols + i]
            length = (vecx*vecx + vecy*vecy)**.5
            theta = math.atan2(vecy , vecx)
            colLength = scale(length , 1 , 15 , 2 , 10)
            col = color(math.log(colLength), math.log(10))
            # col = color(length, 10 )

            arrow(x,y,length , theta , col)
def drawDye() : 
    for j in range(dyeRows) : 
        for i in range(dyeCols) : 
            x , y = i*dyeGap , j*dyeGap
            dyeCol = dye[j*dyeCols + i] * 200
            if dyeCol > 255 : dyeCol = 255
            if dyeCol < 0 : dyeCol = 0
            dyeCol = dyeCol,dyeCol,dyeCol
            pg.draw.rect(screen , dyeCol , pg.Rect(x,y,dyeGap,dyeGap),0)
def fillDye(mx,my) : 
    mouseRad = 50
    xIndex = int(mx/dyeGap)
    yIndex = int(my/dyeGap)
    radUnits = int((mouseRad/dyeGap)/2)
    for j in range(-radUnits , radUnits , 1) : 
        for i in range(-radUnits , radUnits , 1) : 
            iIndex = xIndex + i 
            jIndex = yIndex + j
            xDist = mx - iIndex*dyeGap
            yDist = my - jIndex*dyeGap
            if math.sqrt(xDist*xDist + yDist*yDist) <= mouseRad-20 and 0 < iIndex < dyeCols and 0 < jIndex < dyeRows : 
                index = jIndex*dyeCols + iIndex
                dye[index] = 1 
def fillVels(vecx,vecy , mx , my) : 
    mouseRad = 120
    xIndex = int(mx/gap)
    yIndex = int(my/gap)
    radUnits = int((mouseRad/gap)/2)
    for j in range(-radUnits , radUnits , 1) : 
        for i in range(-radUnits , radUnits , 1) : 
            iIndex = xIndex + i 
            jIndex = yIndex + j
            xDist = mx - iIndex*gap
            yDist = my - jIndex*gap
            if math.sqrt(xDist*xDist + yDist*yDist) <= mouseRad-3*gap and 0 < iIndex < numCols and 0 < jIndex < numRows : 
                if gap < iIndex*gap < width-gap and gap < jIndex*gap < height-gap :
                    index = jIndex*numCols + iIndex
                    u[index] = vecx 
                    v[index] = vecy 
def makeIncompressible(numIter) : 
    global u , v , s 
    for k in range(numIter) :
        for j in range(1,numRows-1) : 
            for i in range(1,numCols-1) : 
                if s[j*numCols + i] == 0 : continue
                sVal = s[j*numCols + (i+1)] + s[j*numCols + (i-1)] + s[(j+1)*numCols + i] + s[(j-1)*numCols + i]
                if sVal == 0 : continue
                div = -u[j*numCols + i] + u[j*numCols + (i+1)] - v[j*numCols + i] + v[(j+1)*numCols + i]
                div = 1.9*div / sVal
                u[j*numCols + i]      += div * s[j*numCols + (i-1)]
                u[j*numCols + (i+1)]  -= div * s[j*numCols + (i+1)]
                v[j*numCols + i]      += div * s[(j-1)*numCols + i]
                v[(j+1)*numCols + i]  -= div * s[(j+1)*numCols + i]
def extrapolate() : 
    global u,v 
    for i in range(numCols) : 
        u[0*numCols + i] = u[1*numCols + i]
        u[(numRows-1)*numCols + i] = u[(numRows-2)*numCols + i]
    for j in range(numRows) :
        v[j*numCols + 0] = v[j*numCols + 1] 
        v[j*numCols + numCols-1] = v[j*numCols + numCols-2] 
def sampleField(x , y , field) : 
    N = numCols
    f = field
    x = max(min(x , numCols*gap) , gap)
    y = max(min(y , numRows*gap) , gap)
    dx = 0
    dy = 0
    if field == "uField" : f , dy = u , gap/2
    elif field == "vField" : f , dx = v , gap/2
    elif field == "sField" : f , dx , dy = dye , gap/2 , gap/2
    x = x - dx 
    y = y - dy 
    i0 = min(int(x/gap) , numCols-1)
    j0 = min(int(y/gap) , numRows-1)
    i1 = i0+1
    j1 = j0+1
    wx1 = (x - i0*gap) / gap
    wy1 = (y - j0*gap) / gap
    wx0 , wy0 = 1 - wx1 , 1 - wy1
    val = wx0*wy0*f[j0*N + i0] + wx1*wy0*f[j0*N + i1] + wx0*wy1*f[j1*N + i0] + wx1*wy1*f[j1*N + i1]
    return val
def avgU(i,j) : 
    N = numCols
    # uVel = u[j*N + i] + u[j*N + i-1] + u[(j+1)*N + i] + u[(j+1)*N + i-1]
    uVel = u[j*N + i] + u[(j-1)*N + i] + u[j*N + i+1] + u[(j-1)*N + i+1]

    uVel = .25 * uVel 
    return uVel 
def avgV(i,j) : 
    N = numCols
    vVel = v[j*N + i] + v[j*N + i-1] + v[(j+1)*N + i] + v[(j+1)*N + i-1]
    vVel = .25 * vVel 
    return vVel 
def advectVel(dt) : 
    global u,v,s
    newU = np.copy(u)
    newV = np.copy(v)
    for j in range(1,numRows) : 
        for i in range(1,numCols) :
            # for u componenet 
            if s[j*numCols + i] != 0 and s[j*numCols + i-1] != 0 and j < numRows-1 : 
                x = i*gap 
                y = j*gap + gap/2
                uVal = u[j*numCols + i]
                vVal = avgV(i,j)
                x = x - uVal * dt 
                y = y - vVal * dt
                uVel = sampleField(x,y,"uField")
                newU[j*numCols + i] = uVel
            # for v component 
            if s[j*numCols + i] != 0 and s[(j-1)*numCols + i] and i < numCols-1 : 
                x = i*gap + gap/2
                y = j*gap 
                uVal = avgU(i,j) 
                vVal = v[j*numCols + i] 
                x = x - uVal * dt 
                y = y - vVal * dt
                vVel = sampleField(x,y,"vField")
                newV[j*numCols + i] = vVel
    u = newU
    v = newV

def advectDye(dt) : 
    global u,v,s,dye
    newDye = np.copy(dye)
    for j in range(1 , dyeRows-1) : 
        for i in range(1, dyeCols-1) : 
            if s[j*numCols + i] != 0 : 
                uVal = (u[j*numCols + i] + u[j*numCols + (i+1)] ) * .5
                vVal = (v[j*numCols + i] + v[(j+1)*numCols + i] ) * .5
                x = (i*gap + gap/2) - uVal*dt
                y = (j*gap + gap/2) - vVal*dt
                newDye[j*numCols + i] = sampleField(x,y,"sField") 
    dye = newDye 

def simulate(numIters , dt) : 
    makeIncompressible(numIters)
    extrapolate()
    advectVel(dt)
    advectDye(dt)

def render(show = True) : 
    # drawS()
    drawDye()
    if show : drawVels()
def setVel(uVel) :
    for j in range(numRows) : 
        for i in range(numCols) :
            if i == 1 : 
                u[j*numCols + i] = uVel


center = width//2 , height//2
p=0
showField = False
changeNoise = True
pmx , pmy = 100,100
mx , my = 100,100

while running : 
    screen.fill((50,50,50))
    mx , my = pg.mouse.get_pos()
    vecx = mx-pmx
    vecy = my-pmy
    mag = vecx*vecx + vecy*vecy
    mag = max(1 , math.sqrt(mag))
    # dye[445] = 1
    simulate(5,1)
    render(showField)
    if pg.mouse.get_pressed()[0] : 
        fillVels(15*vecx/mag,15*vecy/mag , mx , my)
        fillDye(mx,my)
    if pg.mouse.get_pressed()[1] : fillDye(mx,my)
    if pg.mouse.get_pressed()[2] : fillVels(15*vecx/mag,15*vecy/mag , mx , my)




    pmx , pmy = mx , my
    for event in pg.event.get() : 
        if event.type == pg.QUIT : 
            running = False 
        elif event.type == pg.KEYDOWN : 
            if event.key == pg.K_ESCAPE : 
                running = False 
            if event.key == pg.K_SPACE : 
                u = np.zeros(numCells)
                v = np.zeros(numCells)
                dye = np.zeros(dyeCells)
            if event.key == pg.K_p : 
                showField = not showField
    pg.display.flip()
    clock.tick(60)
    pg.display.set_caption(f'FPS : {round(clock.get_fps() , 2)}')


pg.quit()