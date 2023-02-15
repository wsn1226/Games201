import taichi as ti
import random

ti.init(arch = ti.gpu)

Circle_Center = ti.Vector([0.7, 0.2])
Circle_Radius = 0.3

dim = 2
n_grid = 256
dx = 1 / n_grid
inv_dx = 1 / dx
dt = 4.0e-5

p_rho = 1
# Material Parameters
E = 5000 # stretch
gamma = 500 # shear
k = 1000 # normal

# number of lines
N_Line = 12
# line space distance
dlx = 0.009
# type2 particle count per line
ln_type2 = 200
ln_type3 = ln_type2 - 1

start_pos = ti.Vector([0.2, 0.8])

n_type2 = N_Line* ln_type2
n_type3 = N_Line* ln_type3

#line length
Length = 0.75
sl = Length/ (ln_type2-1)

#type2
x2 = ti.Vector.field(2, dtype=float, shape=n_type2) # position 
v2 = ti.Vector.field(2, dtype=float, shape=n_type2) # velocity
C2 = ti.Matrix.field(2, 2, dtype=float, shape=n_type2) # affine velocity field
volume2 =  dx*Length / (ln_type3+ln_type2)

#type3
x3 = ti.Vector.field(2, dtype=float, shape=n_type3) # position
v3 = ti.Vector.field(2, dtype=float, shape=n_type3) # velocity
C3 = ti.Matrix.field(2, 2, dtype=float, shape=n_type3) # affine velocity field
F3 = ti.Matrix.field(2, 2, dtype=float, shape=n_type3) # deformation gradient
D3_inv = ti.Matrix.field(2, 2, dtype=float, shape=n_type3)
d3 = ti.Matrix.field(2, 2, dtype=float, shape=n_type3)
volume3 = volume2

grid_v = ti.Vector.field(2, dtype= float, shape=(n_grid, n_grid))
grid_m = ti.field(dtype=float, shape=(n_grid, n_grid))
grid_f = ti.Vector.field(2, dtype= float, shape = (n_grid, n_grid))

n_segment = n_type3

ROT90 = ti.Matrix([[0,-1.0],[1.0,0]])
print(ROT90)
@ti.func
def QR2(Mat): #2x2 mat, Gram–Schmidt Orthogonalization
    c0 = ti.Vector([Mat[0,0],Mat[1,0]])
    c1 = ti.Vector([Mat[0,1],Mat[1,1]])
    r11 = c0.norm(1e-6)
    q0 = c0/r11
    r12 = c1.dot(q0)
    q1 = c1 - r12 * q0
    r22 = q1.norm(1e-6)
    q1/=r22
    Q = ti.Matrix.cols([q0,q1])
    R = ti.Matrix([[r11,r12],[0,r22]])
    return Q,R


@ti.kernel
def Particle_To_Grid():
    for p in x2:
        base = (x2[p] * inv_dx - 0.5).cast(int)
        fx = x2[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5) ** 2]
        affine = C2[p]
        mass = volume2* p_rho
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i,j])
            weight = w[i][0]*w[j][1]
            grid_m[base + offset] += weight * mass
            dpos = (offset.cast(float) - fx) * dx
            grid_v[base + offset] += weight * mass * (v2[p] +  affine@dpos)

    for p in x3:
        base = (x3[p] * inv_dx - 0.5).cast(int)
        fx = x3[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5) ** 2]
        affine = C3[p]
        mass = volume3* p_rho
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i,j])
            weight = w[i][0]*w[j][1]
            grid_m[base + offset] += weight * mass
            dpos = (offset.cast(float) - fx) * dx
            grid_v[base + offset] += weight * mass * (v3[p] +  affine@dpos)


@ti.kernel
def Grid_Force():
    for p in x3:
        l, n = GetType2FromType3(p)

        base = (x3[p] * inv_dx - 0.5).cast(int)
        fx = x3[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        dw_dx_d = ti.Matrix.rows([fx-1.5, 2*(1.0-fx), fx-0.5]) * inv_dx


        base_l = (x2[l] * inv_dx - 0.5).cast(int)
        fx_l = x2[l] * inv_dx - base_l.cast(float)
        w_l = [0.5 * (1.5 - fx_l) ** 2, 0.75 - (fx_l - 1.0) ** 2, 0.5 * (fx_l - 0.5) ** 2]

        base_n = (x2[n] * inv_dx - 0.5).cast(int)
        fx_n = x2[n] * inv_dx - base_n.cast(float)
        w_n = [ 0.5 * (1.5 - fx_n) ** 2, 0.75 - (fx_n - 1.0) ** 2, 0.5 * (fx_n - 0.5) ** 2]

        Q, R = QR2(F3[p])
        r11 = R[0,0]
        r12 = R[0,1]
        r22 = R[1,1] 
        A = ti.Matrix.rows([[E*r11*(r11-1)+gamma*r12**2, gamma * r12 * r22], \
                        [gamma * r12 * r22,  -k * (1 - r22)**2 * r22 * float(r22 <= 1)]])
        dphi_dF = Q@ A @ R.inverse().transpose()# Q.inverse().transpose() = Q.transpose().transpose() = Q

        dp_c1 = ti.Vector([d3[p][0,1],d3[p][1,1]])

        dphi_dF_c1 = ti.Vector([dphi_dF[0,1],dphi_dF[1,1]])

        Dp_inv_c0 = ti.Vector([D3_inv[p][0,0],D3_inv[p][1,0]])

        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i,j])

            # technical document .(15) part 1
            weight_l = w_l[i][0] * w_l[j][1]
            weight_n = w_n[i][0] * w_n[j][1]
            f_2 = dphi_dF @ Dp_inv_c0
            grid_f[base_l + offset] += volume2 * weight_l * f_2
            grid_f[base_n + offset] += -volume2 * weight_n * f_2

            #dphi w / x
            dw_dx = ti.Vector([ dw_dx_d[i, 0] * w[j][1], w[i][0] * dw_dx_d[j, 1] ])
            # technical document .(15) part 2
            grid_f[base + offset] += -volume3 * dphi_dF_c1* dw_dx.dot( dp_c1 )
    
    # spring force, bending parameter
    for p in range((ln_type2-2)* N_Line):
        nl = p // (ln_type2-2)
        
        v0 = p + nl* 2
        v1 = v0+ 2

        base_0 = (x2[v0] * inv_dx - 0.5).cast(int)
        fx_0 = x2[v0] * inv_dx - base_0.cast(float)
        w_0 = [0.5 * (1.5 - fx_0) ** 2, 0.75 - (fx_0 - 1.0) ** 2, 0.5 * (fx_0 - 0.5) ** 2]

        base_1 = (x2[v1] * inv_dx - 0.5).cast(int)
        fx_1 = x2[v1] * inv_dx - base_1.cast(float)
        w_1 = [0.5 * (1.5 - fx_1) ** 2, 0.75 - (fx_1 - 1.0) ** 2, 0.5 * (fx_1 - 0.5) ** 2]
        

        dir_x = x2[v1] - x2[v0]
        dist = dir_x.norm(1e-9)
        dir_x /= dist
        fn = dist- 2.0*sl
        f = -1000*fn* dir_x

        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i,j])

            weight_0 = w_0[i][0] * w_0[j][1]
            weight_1 = w_1[i][0] * w_1[j][1]
            
            grid_f[base_0 + offset] -= weight_0 * f
            grid_f[base_1 + offset] += weight_1 * f


    # move to Grid_Collision update grid_v
    # for i, j in grid_m:
    #     if grid_m[i, j] > 0:
    #         grid_v[i,j] +=  grid_f[i,j] * dt
    #         grid_v[i, j] /= grid_m[i, j]
    #         grid_v[i, j].y -= dt * 9.80
    
bound = 3
@ti.kernel
def Grid_Collision():
    for i, j in grid_m:
        if grid_m[i, j] > 0:
            # Grid momentum update
            grid_v[i,j] +=  grid_f[i,j] * dt
            grid_v[i, j] /= grid_m[i, j]
            grid_v[i, j].y -= dt * 9.80

            #circle collision
            dist = ti.Vector([i * dx, j * dx]) - Circle_Center
            if dist.x**2 + dist.y**2 < Circle_Radius* Circle_Radius :
                dist = dist.normalized()
                grid_v[i, j] -= dist * min(0, grid_v[i, j].dot(dist) )
                grid_v[i, j] *= 0.9  #friction

            if i < bound and grid_v[i, j].x < 0:
                grid_v[i, j].x = 0
            if i > n_grid - bound and grid_v[i, j].x > 0:
                grid_v[i, j].x = 0
            if j < bound and grid_v[i, j].y < 0:
                grid_v[i, j].y = 0
            if j > n_grid - bound and grid_v[i, j].y > 0:
                grid_v[i, j].y = 0


@ti.kernel
def Grid_To_Particle():
    for p in x2:
        base = (x2[p] * inv_dx - 0.5).cast(int)
        fx = x2[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.Vector.zero(float, 2)
        new_C = ti.Matrix.zero(float, 2, 2)
        for i, j in ti.static(ti.ndrange(3, 3)):
            dpos = ti.Vector([i, j]).cast(float) - fx
            g_v = grid_v[base + ti.Vector([i, j])]
            weight = w[i][0] * w[j][1]
            new_v += weight * g_v
            new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx
        v2[p] = new_v
        x2[p] += dt * v2[p]
        C2[p] = new_C

    for p in x3:
        base = (x3[p] * inv_dx - 0.5).cast(int)
        fx = x3[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_C = ti.Matrix.zero(float, 2, 2)
        for i, j in ti.static(ti.ndrange(3, 3)):
            dpos = ti.Vector([i, j]).cast(float) - fx
            g_v = grid_v[base + ti.Vector([i, j])]
            weight = w[i][0] * w[j][1]
            new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx
        C3[p] = new_C


@ti.kernel
def Update_Particle_State():
    for p in x3:
        l, n = GetType2FromType3(p)
        v3[p] = 0.5 * (v2[l] + v2[n])
        x3[p] = 0.5 * (x2[l] + x2[n])

        dp1 = x2[n] - x2[l]
        dp2 = ti.Vector([d3[p][0,1],d3[p][1,1]])
        dp2 += dt * C3[p]@dp2
        d3[p] = ti.Matrix.cols([dp1,dp2])
        F3[p] = d3[p]@D3_inv[p]


cf = 0.05
@ti.kernel
def Return_Mapping():
    for p in x3:
        Q,R = QR2(F3[p])
        r12 = R[0,1]
        r22 = R[1,1]

        #cf = 0
        if r22 < 0:
            r12 = 0
            r22 = max(r22, -1)
        elif r22> 1:
            r12 = 0
            r22 = 1
        else:
            rr = r12**2
            zz = cf*(1.0 - r22)**2
            gamma_over_s = gamma/k
            f = gamma_over_s**2 * rr - zz**2
            if f > 0:
                scale = zz / ( gamma_over_s*  rr**0.5 )
                r12*= scale

        R[0,1] = r12
        R[1,1] = r22

        F3[p] = Q@R
        d3[p] = F3[p]@D3_inv[p].inverse()



@ti.kernel
def Reset():
    for i, j in grid_m:
        grid_v[i, j] = [0, 0]
        grid_m[i, j] = 0
        grid_f[i, j] = [0.0,0.0]


#get type2 from type3
@ti.func
def GetType2FromType3(index):
    index += index // ln_type3
    return index, index+1

@ti.kernel
def initialize():
    for i in range(n_type2):
        sq = i // ln_type2
        x2[i] = ti.Vector([start_pos[0]+ (i- sq* ln_type2) * sl, start_pos[1] + sq* dlx])
        v2[i] = ti.Matrix([0, 0])
        C2[i] =  ti.Matrix([[0,0],[0,0]])

    for i in range(n_segment):
        l, n = GetType2FromType3(i)

        x3[i] = 0.5*(x2[l] + x2[n])
        v3[i] = ti.Matrix([0, 0])
        F3[i] = ti.Matrix([[1.0, 0.0],[0.0, 1.0] ])
        C3[i] =  ti.Matrix([[0,0],[0,0]])   

        dp0 = x2[n] - x2[l]
        dp1 = ROT90@dp0
        dp1 /= dp1.norm(1e-6)
        d3[i] = ti.Matrix.cols([dp0,dp1])
        D3_inv[i] = d3[i].inverse()     

        


def main():
    initialize()

    randColor = [random.randint(0, 0xffffff) for i in range(N_Line)]

    gui = ti.GUI("Cloth2D", (512, 512))
    while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
        for _ in range(50):
            Reset()
            Particle_To_Grid() # same as before
            Grid_Force()
            Grid_Collision()
            Grid_To_Particle() # same as before
            Update_Particle_State()
            Return_Mapping()

        gui.clear(0x112F41)

        gui.circle(Circle_Center, radius=Circle_Radius* 512, color=0x666666)

        x2_ny = x2.to_numpy()
        for li in range(N_Line):
            gui.circles(x2_ny[li* ln_type2 : (li+1)*ln_type2], radius=2, color= randColor[li])
        gui.show()

if __name__ == "__main__":
    main()