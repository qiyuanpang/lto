import numpy as np
from scipy.special import hankel2

def findge(A,e):
    pos1 = []
    pos2 = []
    n = np.size(A[:,0])
    m = np.size(A[0,:])
    for i in range(0,m):
        for j in range(0,n):
            if A[j,i] >= e:
               pos1.append(j)
               pos2.append(i)
    return pos1, pos2
    
def findeq(A,e):
    pos1 = []
    pos2 = []
    n = np.size(A[:,0])
    m = np.size(A[0,:])
    for i in range(0,m):
        for j in range(0,n):
            if (A[j,i]-e) < 1.0e-16:
               pos1.append(j)
               pos2.append(i)
    return pos1, pos2

def HSSBF_Zfun(N, omega, mu0, dl, k, rho, gamma, sc):
    I = np.reshape(np.array(range(1,N+1)),[N,1])
    J = np.reshape(np.array(range(1,N+1)),[1,N])
    ind = np.abs(np.tile(I,[1,N])-np.tile(J,[N,1]))
    Z = np.zeros([N,N], dtype=complex)
    pos1, pos2 = findge(ind, 1.0)
    #print(pos1)
    I = I-1
    J = J-1
    for i in range(np.size(pos1)):
        #if np.isnan(hankel2(0,k*np.sqrt(np.sum((rho[:,I[pos1[i],0]]-rho[:,J[0,pos2[i]]]))**2))/sc):
        #   print(i,I[pos1[i],0],J[0,pos2[i]],k*np.sqrt(np.sum((rho[:,I[pos1[i],0]]-rho[:,J[0,pos2[i]]]))**2),)
        Z[pos1[i], pos2[i]] = omega*mu0*dl[J[0,pos2[i]]]/4*hankel2(0,k*np.sqrt(np.sum((rho[:,I[pos1[i],0]]-rho[:,J[0,pos2[i]]])**2)))/sc
    pos1, pos2 = findeq(ind, 0.0)
    for i in range(np.size(pos1)):
        Z[pos1[i], pos2[i]] = omega*mu0*dl[J[0,pos2[i]]]/4*(1-1j*(2/np.math.pi)*np.log(gamma*k*dl[J[0,pos2[i]]]/4/np.exp(1.0)))/sc
    
    return Z

def generate_Z(num, N):
    origin = [200,60]
    ppw = 15
    if num == 1:
        ## cycle
        a = 1.0
        b = 1.0
        st = 0
        ed = 2*np.math.pi
        t = np.linspace(st,ed,N)
        dt = (ed-st)/N
        rho1 = np.vstack([a*np.cos(t-dt/2), b*np.sin(t-dt/2)])
        rho2 = np.vstack([a*np.cos(t+dt/2), b*np.sin(t+dt/2)])
        dl = np.sqrt(np.sum((rho2-rho1)**2, axis=0))
        rho = np.vstack([a*np.cos(t), b*np.sin(t)])
    elif num == 2:
        ## spiral lines
        a = 1.0
        b = 1.0
        st = 0
        ed = 2*np.math.pi
        t = np.linspace(st,ed,N)
        dt = (ed-st)/N
        r_st = 1.0
        r_ed = 2.0
        rad = np.linspace(r_st,r_ed,N)
        drad = (r_ed-1)/N
        rho1 = np.vstack([a*np.cos(t-dt/2)*(rad-drad/2), b*np.sin(t-dt/2)*(rad-drad/2)])
        rho2 = np.vstack([a*np.cos(t+dt/2)*(rad+drad/2), b*np.sin(t+dt/2)*(rad+drad/2)])
        dl = np.sqrt(np.sum((rho2-rho1)**2, axis=0))
        rho = np.vstack([a*np.cos(t)*rad, b*np.sin(t)*rad])
    elif num == 3:
        ## half cycle
        a = 1.0
        b = 2.0
        st = 0.0
        ed = np.math.pi
        t = np.linspace(st,ed,N)
        dt = (ed-st)/N
        #print('dt=', dt)
        rho1 = np.vstack([a*np.cos(t-dt/2), b*np.sin(t-dt/2)])
        rho2 = np.vstack([a*np.cos(t+dt/2), b*np.sin(t+dt/2)])
        #print('rho1=',rho1)
        dl = np.sqrt(np.sum((rho2-rho1)**2, axis=0))
        rho = np.vstack([a*np.cos(t), b*np.sin(t)])
    elif num == 4:
        ## spiral lines
        a = 1.0
        b = 1.0
        st = 0
        ed = 0.6*np.math.pi
        t = np.linspace(st,ed,N)
        dt = (ed-st)/N
        r_st = 1.0
        r_ed = 2.0
        rad = np.linspace(r_st,r_ed,N)
        drad = (r_ed-1)/N
        rho1 = np.vstack([a*np.cos(t-dt/2)*(rad-drad/2), b*np.sin(t-dt/2)*(rad-drad/2)])
        rho2 = np.vstack([a*np.cos(t+dt/2)*(rad+drad/2), b*np.sin(t+dt/2)*(rad+drad/2)])
        dl = np.sqrt(np.sum((rho2-rho1)**2, axis=0))
        rho = np.vstack([a*np.cos(t)*rad, b*np.sin(t)*rad])
        
    
    elif num == 5:
        ## parallel strip
        N = N - 1
        rho = np.zeros([2,N])
        rho[0,0:N/2]=-1.0
        rho[1,0:N/2]=np.linspace(-1,1,N/2)
        rho[0,N/2:N]=1.0
        rho[1,N/2:N]=np.linspace(-1,1,N/2)
        dl = np.ones(N)*4/N
        
    elif num == 6:
        ## rectangular cup
        N = N - 1
        rho = np.hstack([np.ones(N/4)*(-2), np.linspace(-2,2,N/2), np.ones(N/4)*2])
        rho = np.vstack([rho, np.hstack([np.linspace(-1,1,N/4), np.zeros(N/2), np.linspace(-1,1,N/4)])])
        dl = np.ones(N)*8/N
        
    elif num == 7:
        ## corrugated line
        a = 1.0
        b = 1.0
        st = 0.0
        ed = np.math.pi
        Nseg=3
        N = N + 2
        t = np.linspace(st,ed,N/Nseg)
        dt = (ed-st)/(N/Nseg)
        rho1 = np.vstack([a*np.cos(t-dt/2), b*np.sin(t-dt/2)])
        rho2 = np.vstack([a*np.cos(t+dt/2), b*np.sin(t+dt/2)])
        dl0 = np.sqrt(np.sum((rho2-rho1)**2, axis=0))
        rho0 = np.vstack([a*np.cos(t), b*np.sin(t)])
        rho = []
        dl=[]
        s=-1.0
        for ii in range(1,2):
            rho0[1,:]=rho0[1,:]*s
            rho0[0,:]=rho0[0,:]+2*a;
            rho = rho0
            dl = dl0
        for ii in range(2,Nseg+1):
            rho0[1,:]=rho0[1,:]*s
            rho0[0,:]=rho0[0,:]+2*a;
            rho = np.hstack([rho,rho0])
            dl = np.hstack([dl, dl0])

    elif num == 8:
        ## kite
        st = 0
        ed = 2*np.math.pi
        t = np.linspace(st,ed,N)
        dt = (ed-st)/N
        a = 2.0+ np.sin(3*t)
        b = 2.0+ np.sin(3*t)
        rho1 = np.vstack([a*np.cos(t-dt/2), b*np.sin(t-dt/2)])
        rho2 = np.vstack([a*np.cos(t+dt/2), b*np.sin(t+dt/2)])
        dl = np.sqrt(np.sum((rho2-rho1)**2, axis=0))
        rho = np.vstack([a*np.cos(t), b*np.sin(t)])
    elif num == 9:
        ## bean
        st = 0.0
        ed = 2*np.math.pi
        t = np.linspace(st,ed,N)
        dt = (ed-st)/N
        a = 2.0+ np.sin(2*t)
        b = 2.0+ np.sin(2*t)
        rho1 = np.vstack([a*np.cos(t-dt/2), b*np.sin(t-dt/2)])
        rho2 = np.vstack([a*np.cos(t+dt/2), b*np.sin(t+dt/2)])
        dl = np.sqrt(np.sum((rho2-rho1)**2, axis=0))
        rho = np.vstack([a*np.cos(t), b*np.sin(t)])
    elif num == 10:
        ## partial kite
        st = 0
        ed = 1.8*np.math.pi
        t = np.linspace(st,ed,N)
        dt = (ed-st)/N
        a = 2.0+ np.sin(3*t)
        b = 2.0+ np.sin(3*t)
        rho1 = np.vstack([a*np.cos(t-dt/2), b*np.sin(t-dt/2)])
        rho2 = np.vstack([a*np.cos(t+dt/2), b*np.sin(t+dt/2)])
        dl = np.sqrt(np.sum((rho2-rho1)**2, axis=0))
        rho = np.vstack([a*np.cos(t), b*np.sin(t)])
    p = np.sum(dl)
    Lambda = p/N*ppw
    k = 2*np.math.pi/Lambda
    mu0 = 4*np.math.pi*1.0e-7
    eps0 = 8.854187*1.0e-12
    c = 1.0/np.sqrt(mu0*eps0)
    omega = 2*np.math.pi*c/Lambda
    gamma = 1.781072418
    sc = 1.0
    if num == 5:
       N = N
    elif num == 6:
       N = N
    elif num == 7:
       N = N-3
    else:
       N = N-1
       
    Z = HSSBF_Zfun(N,omega,mu0,dl,k,rho,gamma,sc)
    nZ = np.max(np.abs(Z[:,0]))
    #print(nZ)
    return Z/nZ
    
'''
def main():
    Z = generate_Z(3, 129)
    U = np.triu(Z)
    L = np.tril(Z,-1)+np.eye(np.size(Z[0,:]))
    #print(np.sum(np.isnan(Z)))
    #print(np.linalg.solve(U,np.eye(np.size(Z[0,:]))))
    E = np.linalg.solve(L,np.matmul(Z,np.linalg.solve(U,np.eye(np.size(Z[0,:])))))
    E = np.real(E)
    Z = np.real(Z)
    print('cond(E) = ', np.linalg.cond(E,2))
    print('cond(Z) = ', np.linalg.cond(Z,2))
    b = np.random.rand(np.size(Z[0,:]))
    lb = np.linalg.solve(L,b)
    y = np.linalg.solve(E,lb)
    x = np.linalg.solve(U,y)
    err = np.linalg.norm(np.matmul(Z,x)-b)/np.linalg.norm(b)
    print('err = ', err)
    
if __name__ == "__main__":
    main()
'''    

