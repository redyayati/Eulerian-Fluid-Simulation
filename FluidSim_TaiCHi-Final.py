import numpy as np
import taichi as ti 
import pygame as pg
ti.init(arch=ti.gpu) 
width = 300
height = 300

screen = pg.display.set_mode((width, height))
pg.display.set_caption('Title')
clock = pg.time.Clock() 
running  = True

cellSize = 5


# Initial Scene Variables
numCols = int(width / cellSize)
numRows = int(height / cellSize )
dye = ti.field(ti.f32, shape=(numCols , numRows))
u = ti.field(ti.f32, shape=(numCols , numRows))
v = ti.field(ti.f32, shape=(numCols , numRows))
newU = ti.field(ti.f32, shape=(numCols , numRows))
newV = ti.field(ti.f32, shape=(numCols , numRows))
newDye = ti.field(ti.f32, shape=(numCols , numRows))
s = ti.field(ti.f32, shape=(numCols , numRows))
image = ti.Vector.field(3, ti.f32, shape=(numCols , numRows))


def swap(field1 , field2) : 
    field1 , field2 = field2 , field1 
@ti.kernel 
def fillS() : 
    for i,j in s : 
        if i == 0  or i == numCols-1  or j == 0  or j==numRows-1  : s[i,j] = 0 
        else : s[i,j] = 1
@ti.func
def divergence(i:int,j:int) : 
    div = -u[i,j] + u[i+1,j] - v[i,j] + v[i,j+1]
    return div
    
@ti.kernel 
def diverge(u : ti.template() , v : ti.template() , s : ti.template()) : 
    div = 0.0
    sVal = 0.0
    for i,j in u :
        if i > 10 and i < 40 and j > 10 and j < 40 : 
            if s[i,j] == 0 : continue
            sVal = s[i+1,j] + s[i-1,j] + s[i,j+1] + s[i,j-1]
            if sVal == 0 : continue 
            # div = divergence(i,j)
            div = -u[i,j] + u[i+1,j] - v[i,j] + v[i,j+1]
            div = 1.9*div/sVal 
            u[i,j] += div * s[i-1,j] 
            u[i+1,j] -= div * s[i+1,j] 
            v[i,j] += div * s[i, j-1]
            v[i,j+1] -= div * s[i,j+1]

@ti.kernel 
def newDiverge(u : ti.template() , v : ti.template() , s : ti.template()) : 
    for i,j in u :
        if i > 0 and i < numCols-1 and j > 0 and j < numRows-1 : 
            if s[i,j] == 0 : continue
            sVal = s[i+1,j] + s[i-1,j] + s[i,j+1] + s[i,j-1]
            if sVal == 0 : continue 
            div = -u[i,j] + u[i+1,j] - v[i,j] + v[i,j+1]
            div = 1.9*div/sVal 
            u[i,j] += div * s[i-1,j] 
    for i,j in u :                     
        if i > 0 and i < numCols-1 and j > 0 and j < numRows-1 : 
                    if s[i,j] == 0 : continue
                    sVal = s[i+1,j] + s[i-1,j] + s[i,j+1] + s[i,j-1]
                    if sVal == 0 : continue 
                    div = -u[i,j] + u[i+1,j] - v[i,j] + v[i,j+1]
                    div = 1.9*div/sVal 
                    u[i+1,j] -= div * s[i+1,j] 
    for i,j in u :                 
        if i > 0 and i < numCols-1 and j > 0 and j < numRows-1 : 
                    if s[i,j] == 0 : continue
                    sVal = s[i+1,j] + s[i-1,j] + s[i,j+1] + s[i,j-1]
                    if sVal == 0 : continue 
                    div = -u[i,j] + u[i+1,j] - v[i,j] + v[i,j+1]
                    div = 1.9*div/sVal 
                    v[i,j] += div * s[i, j-1]
    for i,j in u :                
        if i > 0 and i < numCols-1 and j > 0 and j < numRows-1 : 
                    if s[i,j] == 0 : continue
                    sVal = s[i+1,j] + s[i-1,j] + s[i,j+1] + s[i,j-1]
                    if sVal == 0 : continue 
                    div = -u[i,j] + u[i+1,j] - v[i,j] + v[i,j+1]
                    div = 1.9*div/sVal 
                    v[i,j+1] -= div * s[i,j+1]

@ti.kernel 
def extrapolate(u : ti.template() , v : ti.template()) : 
    for i,j in u : 
        u[i,0] = u[i,1]
        u[i,numRows-1] = u[i,numRows-2]
        v[0,j] = v[1,j]
        v[numCols-1,j] = v[numCols-2 ,j]
@ti.func
def sampleField(x : ti.f32 ,y : ti.f32 ,field : int) -> ti.f32 : 
    val = 0.0
    dx , dy = 0.0 , 0.0
    x = ti.max(ti.min(x , width) , cellSize)
    y = ti.max(ti.min(y , height) , cellSize)
    if field == 0 :   dy = cellSize/2
    elif field == 1 : dx = cellSize/2
    elif field == 2 : dx , dy =  cellSize/2 , cellSize/2
    x -= dx 
    y -= dy 
    i0 = ti.min(int(x/cellSize) , numCols-1)
    j0 = ti.min(int(y/cellSize) , numRows-1)
    i1 = i0+1
    j1 = j0+1
    wx1 = (x - i0*cellSize) / cellSize
    wy1 = (y - j0*cellSize) / cellSize
    wx0 , wy0 = 1 - wx1 , 1 - wy1
    if field == 0: 
        val = wx0*wy0*u[i0,j0] + wx1*wy0*u[i1,j0] + wx0*wy1*u[i0,j1] + wx1*wy1*u[i1,j1]
    elif field == 1 : 
        val = wx0*wy0*v[i0,j0] + wx1*wy0*v[i1,j0] + wx0*wy1*v[i0,j1] + wx1*wy1*v[i1,j1]
    elif field == 2 :
        val = wx0*wy0*dye[i0,j0] + wx1*wy0*dye[i1,j0] + wx0*wy1*dye[i0,j1] + wx1*wy1*dye[i1,j1]
    return val
@ti.func
def avgU(i,j) : 
    uVel = u[i,j] + u[i,j-1] + u[i+1,j] + u[i+1,j-1]
    return .25 * uVel 
@ti.func
def avgV(i,j):
    vVel = v[i,j] + v[i-1,j] + v[i,j+1] + v[i-1,j+1]
    return .25 * vVel 
@ti.kernel
def advectVels(dt : ti.f32, u : ti.template() , v : ti.template() , newU : ti.template() , newV : ti.template()) :
    for i , j in u : 
        if i > 0 and j > 0 : 
            # for u component
            if s[i,j] != 0 and s[i-1 , j] != 0 and j < numRows-1 : 
                x , y = i * cellSize , j * cellSize + cellSize/2
                uVal = u[i,j] 
                vVal = avgV(i,j) 
                x -= uVal * dt 
                y -= vVal * dt
                newU[i,j] = sampleField(x,y,0)
            # for v componenet 
            if s[i,j] != 0 and s[i,j-1] != 0 and i < numCols-1 : 
                x , y = i * cellSize + cellSize/2, j * cellSize 
                uVal = avgU(i,j)
                vVal = v[i,j]
                x -= uVal * dt 
                y -= vVal * dt 
                newV[i,j] = sampleField(x,y,1) 
@ti.kernel 
def advectDye(dt : ti.f32, newDye:ti.template() ): 
    for i,j in dye : 
        if s[i,j] != 0 : 
            uVal = (u[i,j] + u[i+1 , j]) * .5
            vVal = (v[i,j] + v[i , j+1]) * .5
            x = (i*cellSize + cellSize/2) - uVal*dt 
            y = (j*cellSize + cellSize/2) - vVal*dt 
            newDye[i,j] = sampleField(x,y,2) 
    
@ti.kernel
def paintDye(): 
    for i,j in image : 
        image[i,j] = dye[i,j],dye[i,j],dye[i,j]



def renderDye(): 
    dyeNP = dye.to_numpy()
    # dyeNP = dyeNP.flatten('F')
    for j in range(numRows) : 
        for i in range(numCols) : 
            x , y = i*cellSize , j*cellSize
            # col = dyeNP[j*numCols + i] 
            col = dyeNP[i,j]
            dyeCol = (col , col , col )
            pg.draw.rect(screen , dyeCol , pg.Rect(x,y,cellSize , cellSize),0) 
def renderVels() : 
    uNP = u.to_numpy()
    vNP = v.to_numpy()
    # uNP = uNP.flatten('F')
    # vNP = vNP.flatten('F')
    for j in range(numRows) : 
        for i in range(numCols) : 
            x , y   = i * cellSize , j * cellSize
            x1 , y1 = x + uNP[i,j]/2 , y + vNP[i,j]/2 
            # x1 , y1 = x + uNP[j*numCols + i] , y + vNP[j*numCols + i] 
            col     = (255,255,255)
            pg.draw.line(screen , col , (x,y) , (x1,y1) , 1)
@ti.kernel
def fillDyeTai(mx:ti.f32,my:ti.f32) : 
    rad = 30
    x = int(mx/cellSize)
    y = int(my/cellSize) 
    rad = int(rad / cellSize)
    for j in range(-rad , rad) : 
        for i in range(-rad , rad) : 
            iIndx = x + i 
            jIndx = y + j 
            if s[iIndx,jIndx] != 0 and 1 < iIndx < numCols-1 and 1 < jIndx < numRows-1: 
                dye[iIndx , jIndx] = 200
@ti.kernel
def fillVelsTai(mx:ti.f32,my:ti.f32 , vecx:ti.f32 , vecy:ti.f32) : 
    rad = 30
    x = int(mx/cellSize)
    y = int(my/cellSize) 
    rad = int(rad / cellSize)
    for j in range(-rad , rad) : 
        for i in range(-rad , rad) : 
            iIndx = x + i 
            jIndx = y + j 
            if s[iIndx,jIndx] != 0 and 1 < iIndx < numCols-1 and 1 < jIndx < numRows-1: 
                u[iIndx , jIndx] = vecx
                v[iIndx , jIndx] = vecy
def step(dt) : 
    global dye , newDye , u , v , newU , newV
    for i in range(40) : newDiverge(u ,v , s) 
    extrapolate(u,v)
    advectVels(dt , u ,v , newU , newV)
    u , newU = newU , u
    v , newV = newV , v
    advectDye(dt , newDye)
    dye , newDye = newDye , dye
fillS()
pmx , pmy = 100,100
mx , my = 100,100
force = 10
sNP = s.to_numpy()
def renderS() : 
    global sNP 
    for j in range(numRows) : 
        for i in range(numCols) : 
            x , y = i*cellSize , j*cellSize
            # col = dyeNP[j*numCols + i] 
            col = sNP[i,j] * 255 
            dyeCol = (0 , col , col )
            pg.draw.rect(screen , dyeCol , pg.Rect(x,y,cellSize , cellSize),0) 
showVels = False
print(numCols*numRows)
while running : 
    screen.fill((50,50,50))

    mx , my = pg.mouse.get_pos()
    vecx = mx-pmx
    vecy = my-pmy
    mag = vecx*vecx + vecy*vecy
    mag = max(1 , ti.sqrt(mag))
    renderDye()
    step(.5)
    # renderS()
    if showVels : renderVels()
    if pg.mouse.get_pressed()[0] : 
        fillVelsTai(mx,my , force*vecx/mag , force*vecy/mag)
        fillDyeTai(mx,my)
    if pg.mouse.get_pressed()[1] : fillDyeTai(mx,my) 
    if pg.mouse.get_pressed()[2] : 
        fillVelsTai(mx,my , force*vecx/mag , force*vecy/mag)
    pmx,pmy = mx,my


    # paintDye()
    # dyePG = image.to_numpy()
    # pg.surfarray.blit_array(screen , dyePG)
    
    for event in pg.event.get() : 
        if event.type == pg.QUIT : 
            running = False 
        elif event.type == pg.KEYDOWN : 
            if event.key == pg.K_ESCAPE : 
                running = False 
            if event.key == pg.K_p : 
                showVels = not showVels
            if event.key == pg.K_SPACE : 
                dye.fill(0)
                u.fill(0)
                v.fill(0)
    pg.display.flip()
    clock.tick(60)
    pg.display.set_caption(f' {round(clock.get_fps() , 2)}')
pg.quit()