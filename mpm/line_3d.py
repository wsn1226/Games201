import taichi as ti
import random

ti.init(arch = ti.gpu)

Ball_Center = ti.Vector.field(3, dtype=float, shape=(1, ))
Ball_Center[0] = [0.7, 0.2, 0.3]
Ball_Radius = 0.3

dim = 3
n_grid = 256
dx = 1 / n_grid
inv_dx = 1 / dx
dt = 4.0e-5

p_rho = 1
# Material Parameters
E = 500 # stretch
gamma = 500 # shear
k = 4000 # normal
beta = 0.8390996312
alpha = 0.445300464

nu_s = 0.3 # sand's Young's modulus and Poisson's ratio
mu_s, lambda_s = E / (2 * (1 + nu_s)), E * nu_s / ((1 + nu_s) * (1 - 2 * nu_s)) # sand's Lame parameters


# number of lines
N_Line = 12
# line space distance
dlx = 0.009
# type2 particle count per line
ln_type2 = 200
ln_type3 = ln_type2 - 1

start_pos = ti.Vector([0.2, 0.8, 0.2])

n_type2 = N_Line* ln_type2
n_type3 = N_Line* ln_type3

#line length
Length = 0.75
sl = Length/ (ln_type2-1)

#type2
x2 = ti.Vector.field(3, dtype=float, shape=n_type2) # position 
v2 = ti.Vector.field(3, dtype=float, shape=n_type2) # velocity
C2 = ti.Matrix.field(3, 3, dtype=float, shape=n_type2) # affine velocity field
volume2 =  dx*Length / (ln_type3+ln_type2)

#type3
x3 = ti.Vector.field(3, dtype=float, shape=n_type3) # position
v3 = ti.Vector.field(3, dtype=float, shape=n_type3) # velocity
C3 = ti.Matrix.field(3, 3, dtype=float, shape=n_type3) # affine velocity field
F3 = ti.Matrix.field(3, 3, dtype=float, shape=n_type3) # deformation gradient
D3_inv = ti.Matrix.field(3, 3, dtype=float, shape=n_type3)
d3 = ti.Matrix.field(3, 3, dtype=float, shape=n_type3)
volume3 = volume2

grid_v = ti.Vector.field(3, dtype= float, shape=(n_grid, n_grid, n_grid))
grid_m = ti.field(dtype=float, shape=(n_grid, n_grid, n_grid))
grid_f = ti.Vector.field(3, dtype= float, shape = (n_grid, n_grid, n_grid))

n_segment = n_type3

ROT90 = ti.Matrix([[0, -1.0, 0],[1.0, 0, 0], [0, 0, 1.0]])

@ti.func
def QR3(Mat): #2x2 mat, Gram–Schmidt Orthogonalization
    c0 = ti.Vector([Mat[0,0],Mat[1,0], Mat[2,0]])
    c1 = ti.Vector([Mat[0,1],Mat[1,1], Mat[2,1]])
    c2 = ti.Vector([Mat[0,2],Mat[1,2], Mat[2,2]])
    r11 = c0.norm(1e-6)
    q0 = c0/r11

    r12 = c1.dot(q0)
    q1 = c1 - r12 * q0
    r22 = q1.norm(1e-6)
    q1/=r22

    r13 = c2.dot(q0)
    r23 = c2.dot(q1)
    q2 = c2 - r13*q0 - r23*q1
    r33 = q2.norm(1e-6)
    q2/=r33
    Q = ti.Matrix.cols([q0,q1,q2])
    R = ti.Matrix([[r11,r12,r23],[0,r22,r23],[0,0,r33]])
    return Q,R


@ti.func
def project(FE):
    U, sig, V = ti.svd(FE)
    e = ti.Matrix([[ti.log(sig[0, 0]), 0], [0, ti.log(sig[1, 1])]])
    ehat = e - e.trace() / 2 * ti.Matrix.identity(float, 2)
    Fnorm = ti.sqrt(ehat[0, 0] ** 2 + ehat[1, 1] ** 2) # Frobenius norm
    yp = Fnorm + (2 * lambda_s + 2 * mu_s) / (2 * mu_s) * e.trace() * alpha # delta gamma
    new_e = ti.Matrix.zero(float, 2, 2)
    if Fnorm <= 0 or e.trace() > 0: # Case II:
        new_e = ti.Matrix.zero(float, 2, 2)
    elif yp <= 0: # Case I:
        new_e = e # return initial matrix without volume correction and cohesive effect
    else: # Case III:
        new_e = e - yp / Fnorm * ehat
    
    Hp = e - yp/Fnorm*e
    tor = 2*mu_s*Hp + lambda_s*Hp.trace()*ti.Matrix.identity(float, 2)
    trace_tor = tor.trace()
    return trace_tor

@ti.kernel
def Particle_To_Grid():
    for p in x2:
        base = (x2[p] * inv_dx - 0.5).cast(int)
        fx = x2[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5) ** 2]
        affine = C2[p]
        mass = volume2* p_rho
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i,j,k])
            weight = w[i][0]*w[j][1]* w[k][2]
            grid_m[base + offset] += weight * mass
            dpos = (offset.cast(float) - fx) * dx
            grid_v[base + offset] += weight * mass * (v2[p] +  affine@dpos)

    for p in x3:
        base = (x3[p] * inv_dx - 0.5).cast(int)
        fx = x3[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5) ** 2]
        affine = C3[p]
        mass = volume3* p_rho
        for i, j,k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i,j,k])
            weight = w[i][0]*w[j][1]* w[k][2]
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

        Q, R = QR3(F3[p])
        r11, r12, r13, r22, r23, r33 = R[0,0], R[0,1], R[0,2], R[1,1], R[1,2], R[2,2]
        ep1 = ti.log(r22)
        ep2 = ti.log(r33)
        A = ti.Matrix.rows([[E*r11*(r11-1)+gamma*(r12**2+r13**2), gamma * (r12 * r22 + r13*r23), gamma*r13*r33  ], \
                        [gamma * (r12 * r22 + r13*r23),             1/r22*(2.0*mu_s*ep1+lambda_s*(ep1+ep2)), 0.0], \
                        [gamma*r13*r33,                             0.0, 1/r22*(2.0*mu_s*ep2+lambda_s*(ep1+ep2))]])
        dphi_dF = Q@ A @ R.inverse().transpose()# Q.inverse().transpose() = Q.transpose().transpose() = Q
        dp_c1 = ti.Vector([d3[p][0,1],d3[p][1,1],d3[p][2,1]])
        dp_c2 = ti.Vector([d3[p][0,2],d3[p][1,2],d3[p][2,2]])

        dphi_dF_c1 = ti.Vector([dphi_dF[0,1],dphi_dF[1,1],dphi_dF[2,1]])
        dphi_dF_c2 = ti.Vector([dphi_dF[0,2],dphi_dF[1,2],dphi_dF[2,2]])

        Dp_inv_c0 = ti.Vector([D3_inv[p][0,0],D3_inv[p][1,0],D3_inv[p][2,0]])

        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i,j, k])

            # technical document .(15) part 1
            weight_l = w_l[i][0] * w_l[j][1] * w_l[k][2]
            weight_n = w_n[i][0] * w_n[j][1] * w_n[k][2]
            f_2 = dphi_dF @ Dp_inv_c0
            grid_f[base_l + offset] += volume2 * weight_l * f_2
            grid_f[base_n + offset] += -volume2 * weight_n * f_2
            #dphi w / x
            dw_dx = ti.Vector([ dw_dx_d[i, 0] * w[j][1], w[i][0] * dw_dx_d[j, 1], w[k][2] * dw_dx_d[k, 2] ])
            # technical document .(15) part 2
            grid_f[base + offset] += -volume3 * dphi_dF_c1* dw_dx.dot( dp_c1 )
            grid_f[base + offset] += -volume3 * dphi_dF_c2* dw_dx.dot( dp_c2 )
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

        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i,j, k])

            weight_0 = w_0[i][0] * w_0[j][1] * w_0[k][2]
            weight_1 = w_1[i][0] * w_1[j][1] * w_1[k][2]
            
            grid_f[base_0 + offset] -= weight_0 * f
            grid_f[base_1 + offset] += weight_1 * f

    # move to Grid_Collision update grid_v
    # for i, j in grid_m:
    #     if grid_m[i, j] > 0:
    #         grid_v[i,j] +=  grid_f[i,j] * dt
    #         grid_v[i, j, k] /= grid_m[i, j]
    #         grid_v[i, j, k].y -= dt * 9.80
    
bound = 3
@ti.kernel
def Grid_Collision():
    for i, j,k in grid_m:
        if grid_m[i, j, k] > 0:
            grid_v[i, j, k] +=  grid_f[i, j, k] * dt
            grid_v[i, j, k] /= grid_m[i, j, k]
            grid_v[i, j, k].y -= dt * 20

            #Ball collision
            dist = ti.Vector([i * dx, j * dx, k * dx]) - Ball_Center[0]
            if dist.x**2 + dist.y**2 + dist.z**2 < Ball_Radius* Ball_Radius :
                dist = dist.normalized()
                grid_v[i, j, k] -= dist * min(0, grid_v[i, j, k].dot(dist) )
                grid_v[i, j, k] *= 0.9  #friction

            if i < bound and grid_v[i, j, k].x < 0:
                grid_v[i, j, k].x = 0
            if i > n_grid - bound and grid_v[i, j, k].x > 0:
                grid_v[i, j, k].x = 0
            if j < bound and grid_v[i, j, k].y < 0:
                grid_v[i, j, k].y = 0
            if j > n_grid - bound and grid_v[i, j, k].y > 0:
                grid_v[i, j, k].y = 0
            if k < bound and grid_v[i, j, k].z < 0:
                grid_v[i, j, k].z = 0
            if k > n_grid - bound and grid_v[i, j, k].z > 0:
                grid_v[i, j, k].z = 0


@ti.kernel
def Grid_To_Particle():
    for p in x2:
        base = (x2[p] * inv_dx - 0.5).cast(int)
        fx = x2[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.Vector.zero(float, 3)
        new_C = ti.Matrix.zero(float, 3, 3)
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            dpos = ti.Vector([i, j, k]).cast(float) - fx
            g_v = grid_v[base + ti.Vector([i, j, k])]
            weight = w[i][0] * w[j][1] * w[k][2]
            new_v += weight * g_v
            new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx
        v2[p] = new_v
        x2[p] += dt * v2[p]
        C2[p] = new_C

    for p in x3:
        base = (x3[p] * inv_dx - 0.5).cast(int)
        fx = x3[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_C = ti.Matrix.zero(float, 3, 3)
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            dpos = ti.Vector([i, j, k]).cast(float) - fx
            g_v = grid_v[base + ti.Vector([i, j, k])]
            weight = w[i][0] * w[j][1] * w[k][2]
            new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx
        C3[p] = new_C


@ti.kernel
def Update_Particle_State():
    for p in x3:
        l, n = GetType2FromType3(p)
        v3[p] = 0.5 * (v2[l] + v2[n])
        x3[p] = 0.5 * (x2[l] + x2[n])

        dp1 = x2[n] - x2[l]
        dp2 = ti.Vector([d3[p][0,1],d3[p][1,1],d3[p][2,1]])
        dp3 = ti.Vector([d3[p][0,2],d3[p][1,2],d3[p][2,2]])
        dp2 += dt * C3[p]@dp2
        dp3 += dt * C3[p]@dp3
        d3[p] = ti.Matrix.cols([dp1,dp2,dp3])
        F3[p] = d3[p]@D3_inv[p]


cf = 0.05
@ti.kernel
def Return_Mapping():
    for p in x3:
        Q, R = QR3(F3[p])
        r11, r12, r13, r22, r23, r33 = R[0,0], R[0,1], R[0,2], R[1,1], R[1,2], R[2,2]
        Rhat3 = ti.Matrix.rows([[r22, r23],[0, r33]])
        trace_tor_div2 = project(Rhat3) / 2.0 #p
        #cf = 0
        r22 = trace_tor_div2
        r33 = trace_tor_div2
        a = r22*r12 + r23*r13
        b = r33 * r13
        length = ti.sqrt(a**2 + b**2)
        k = (trace_tor_div2*cf/length)

        R[0,1] = r12 * k
        R[0,2] = r13 * k

        F3[p] = Q@R
        d3[p] = F3[p]@D3_inv[p].inverse()

@ti.kernel
def Reset():
    for i, j, k in grid_m:
        grid_v[i, j, k] = [0, 0, 0]
        grid_m[i, j, k] = 0
        grid_f[i, j, k] = [0.0,0.0,0.0]


#get type2 from type3
@ti.func
def GetType2FromType3(index):
    index += index // ln_type3
    return index, index+1

@ti.kernel
def initialize():
    for i in range(n_type2):
        sq = i // ln_type2
        x2[i] = ti.Vector([start_pos[0]+ (i- sq* ln_type2) * sl, start_pos[1] + sq* dlx, start_pos[2] + sq*dlx])
        v2[i] = ti.Matrix([0, 0, 0])
        C2[i] =  ti.Matrix([[0,0,0],[0,0,0], [0,0,0]])

    for i in range(n_segment):
        l, n = GetType2FromType3(i)

        x3[i] = 0.5*(x2[l] + x2[n])
        v3[i] = ti.Matrix([0, 0, 0])
        F3[i] = ti.Matrix([[1.0, 0.0, 0.0],[0.0, 1.0, 0.0],[0.0, 0.0, 1.0] ])
        C3[i] =  ti.Matrix([[0,0,0],[0,0,0],[0,0,0]])   

        dp0 = x2[n] - x2[l]
        dp1 = ROT90@dp0
        dp1 /= dp1.norm(1e-6)
        dp2 = dp0.cross(dp1)
        dp2 /= dp2.norm(1e-6)
        d3[i] = ti.Matrix.cols([dp0,dp1,dp2])
        D3_inv[i] = d3[i].inverse()     

        


initialize()

randColor = [random.randint(0, 0xffffff) for i in range(N_Line)]

window = ti.ui.Window("Cloth3D", (1024, 1024),vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()

while window.running:
    for i in range(50):
        Reset()
        Particle_To_Grid() # same as before
        Grid_Force()
        Grid_Collision()
        Grid_To_Particle() # same as before
        Update_Particle_State()
        Return_Mapping()

    camera.position(0.0, 0.0, 3)
    camera.lookat(0.0, 0.0, 0)
    scene.set_camera(camera)

    scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.particles(x2, radius=0.01, color= (0.5, 0.42, 0.8))
    scene.particles(Ball_Center, radius=Ball_Radius * 0.95, color=(0.5, 0.42, 0.8))
    canvas.scene(scene)
    window.show()


""" Q, R = QR3(F3[p]@D3_inv[p].inverse())
r11, r12, r13, r22, r23, r33 = R[0,0], R[0,1], R[0,2], R[1,1], R[1,2], R[2,2]
ep1 = ti.log(r22)
ep2 = ti.log(r33)

A = ti.Matrix.rows([[E*r11*(r11-1)+gamma*(r12**2+r13**2), gamma * (r12 * r22 + r13*r23), gamma*r13*r33  ], \
                [gamma * (r12 * r22 + r13*r23),             1/r22*(2.0*mu_s*ep1+lambda_s*(ep1+ep2)), 0.0], \
                [gamma*r13*r33,                             0.0, 1/r22*(2.0*mu_s*ep2+lambda_s*(ep1+ep2))]])
dphi_dF = Q@ A @ R.inverse().transpose()# Q.inverse().transpose() = Q.transpose().transpose() = Q
cauchy_stress = 1/F3[p].determinant() * dphi_dF @ F3[p].transpose()
#cf = 0
cauchy_Qbasis = Q.transpose()@cauchy_stress@Q
s22,s33,s23,s12,s13 = cauchy_Qbasis[1,1], cauchy_Qbasis[2,2],cauchy_Qbasis[1,2],cauchy_Qbasis[0,1],cauchy_Qbasis[0,2]
J2 = (s22-s33)**2 + 4.0*s23
#R3hat = ti.Matrix.rows([[1.0,0.0,0.0],[0.0,r22,r23],[0.0,0.0,r33]])
if ep1+ep2>=0:
    ep1=0
    ep2=0
elif (ti.sqrt(J2)+alpha/2*(s22+s33)<=0):
    ep1=ep1
    ep2=ep2
else:
    yita = (ep1-ep2)/2+alpha*lambda_s*(ep1+ep2)/(4*mu_s)
    ep1-=yita
    ep2-=yita

if ti.sqrt(s12**2+s13**2)+beta/2*(s22+s33)>0:
    coe = -0.5*beta*(s22+s33)/ti.sqrt(s12**2+s13**2)
    r12*=coe
    r13*=coe

r22 = ti.exp(ep1)
r33 = ti.exp(ep2)

R[0,1] = r12
R[0,2] = r13
R[1,1] = r22
R[2,2] = r33

F3[p] = Q@R@D3_inv[p] """