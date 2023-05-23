import numpy as np
import matplotlib.pyplot as plt
import time
import imageio.v2 as imageio
# from multiprocessing import Pool
from matplotlib.colors import ListedColormap






# Calculation of the Laplacian with periodic boundary condition
#       using Euler discretization method with 4 nearest neighbour interactions.      
def del_sq(phi):
    phi_a = np.roll(phi,1,axis = 0)
    # phi_a[:,0] = phi[:,0]
    phi_b = np.roll(phi,-1,axis = 0)
    # phi_b[:,-1] = phi[:,-1]
    phi_c = np.roll(phi,1,axis = 1)
    # phi_c[0,:] = phi[0,:]
    phi_d = np.roll(phi,-1,axis = 1)
    # phi_d[-1,:] = phi[-1,:]
    
    dx = 1
    grad = (phi_a + phi_b + phi_c + phi_d - 4*phi)/dx**2
    
    return grad




#  The TDGL equation
def TDGL(phi):
    t_len = 3500
    dt = 0.01
    data = np.zeros((len(phi),len(phi),t_len))
    for i in range(t_len):
        phi += (phi - phi**3 + del_sq(phi))*dt
        
        data[:,:,i] = phi
    return data



#  The CHC equation
def CHC(psi):
    t_len = 4000
    dt = 0.01
    data = np.zeros((len(psi),len(psi),t_len))
    for i in range(t_len):
        phi = psi**3 - psi - del_sq(psi)
        psi += del_sq(phi)*dt
        
        data[:,:,i] = psi
    return data





##................... Execution of ESN ................

if __name__ == '__main__':
    start_time = time.time()
    
    
######## Define here the system (TDGL/CHC) for which the ESN to be excecuted ###########
    model = TDGL #CHC
    
    
########  Define the parameter/hyper-parameters ################  

    l = 128             #Lattice size
    dim = 5             #Dimension of input
    icn = 3             #Number of training time series

    N = 100             #reservoir size
    res_rho = 0.7       #Spectral radius
    res_k = 20          #Reservoir degree of connection
    W_in_a = 0.1        # Range of input connection
    alpha = 0.3         #Leaking parameter
    beta = 1e-6         #Regularization parameter

    train_len = 2000           #training length
    pred_len = 1001            #predicting length
    trans = 100                #abondon reservoir initial reservoir evolution
    
    
    
    
    
    
    
    

    

#..................Making Reservoir....................
    print('Generating reservoir')
    W = np.zeros((N,N))
    pr = res_k/(N-1)
    
    for i in range(N):
        for j in range(N):
            if((i!=j) and (np.random.random()<pr)):
                W[i,j] = np.random.random()
                
    eig_val,eig_vec = np.linalg.eig(W)
    m = np.abs(eig_val).max()
    
    W = (res_rho/m)*W
    
    print("Done\n")
    
    
#...................Input connection....................
    
    W_in = np.zeros((N,dim))
    
    for i in range(N):
        W_in[i,int(i*(dim-1)/N)] = W_in_a*(2*np.random.random() - 1)
        
    W_in[:,-1] = W_in_a*(2*np.random.random() - 1)
    
    

#................Training the Reservoir................
    
    store = train_len - trans
    R1 = np.zeros((N,icn*store))
    T1 = np.zeros(icn*store)
    
    
    mix = 0.5
    lim = 0.05
    
    for ic in range(icn):
        psi = np.random.random((l,l))
        psi = np.where(psi<mix,lim*psi/mix,-lim*(psi-mix)/(1-mix))
        
        print("Preparing training data")
        data = model(psi)
        print("Done\n") 
        
        R = np.zeros((N,store))
        
        x = np.zeros(N)
        xt = np.zeros(N)
            
        rw = int(l/2)
        cl = int(l/2)
        for i in range(train_len):
            u = np.array([data[rw-1,cl,i],data[(rw+1)%l,cl,i],data[rw,cl-1,i],data[rw,(cl+1)%l,i],data[rw,cl,i]])
            # u = np.array([data[rw-1,cl,i],data[(rw+1)%l,cl,i],data[rw,cl-1,i],data[rw,(cl+1)%l,i],data[rw,cl,i],data[rw-1,cl-1,i],data[rw-1,(cl+1)%l,i],data[(rw+1)%l,cl-1,i],data[(rw+1)%l,(cl+1)%l,i]])
            x = (1 - alpha)*x + alpha*np.tanh(np.dot(W_in,u) + np.dot(W, x))
            xt[:] = x[:]
            xt[::2] = x[::2]**2
                
                
            if(i>=trans):
                R[:,i-trans] = xt[:]
                
            prog = i*100/train_len
            if(prog == int(prog)):
                print("Training the reservoir: %d %%" %(int(prog)),end = '\r')
            
        
        R1[:,ic*store:(ic+1)*store] = R
        T1[ic*store:(ic+1)*store] = data[rw,cl,trans+1:train_len+1]
        print("\nTraining completed for set %d\n"%(ic+1))
    
    
#..............................Regression............................ 

    print("Regression in progress\n")
        
    W_out = np.dot(np.dot(T1,R1.T),np.linalg.inv((np.dot(R1,R1.T)+beta*np.identity(N))))
        
        
#.........................Predicting Phase.......................


    psi = np.random.random((l,l))
    psi = np.where(psi<mix,lim*psi/mix,-lim*(psi-mix)/(1-mix))
    
    print("Preparing testing data")
    data_T = model(psi)
    print("Done\n")            
    
    
    pred = np.zeros((l,l,pred_len))
    
    m = 200 #np.random.randint(500)    
    pred_nxt = data_T[:,:,m]
    
    warmup = 100
    
    x = np.zeros((N,l*l))
    xt = np.zeros((N,l*l))
    
    prog1 = 0
    for i in range(pred_len+warmup):
        for j in range(l*l):
            rw = int(j/l)
            cl = j % l        
            ut = np.array([pred_nxt[rw-1,cl],pred_nxt[(rw+1)%l,cl],pred_nxt[rw,cl-1],pred_nxt[rw,(cl+1)%l],pred_nxt[rw,cl]])
            # ut = np.array([pred_nxt[rw-1,cl],pred_nxt[(rw+1)%l,cl],pred_nxt[rw,cl-1],pred_nxt[rw,(cl+1)%l],pred_nxt[rw,cl],pred_nxt[rw-1,cl-1],pred_nxt[rw-1,(cl+1)%l],pred_nxt[(rw+1)%l,cl-1],pred_nxt[(rw+1)%l,(cl+1)%l]])            
            x[:,j] = (1 - alpha)*x[:,j] + alpha*np.tanh(np.dot(W_in,ut) + np.dot(W, x[:,j]))
            xt[:,j] = x[:,j]
            xt[::2,j] = x[::2,j]**2
            
            if(i>=warmup):
                pred_nxt1 = np.dot(W_out,xt[:,j])
                pred[rw,cl,i-warmup] = pred_nxt1
            
        if(i>=warmup):
            pred_nxt = pred[:,:,i-warmup]
        
        prog = int(i*100/(pred_len+warmup))
        if(prog != prog1):
            print("Running prediction: %d %%" %(int(prog)),end = '\r')
            prog1 = prog
    
    
    org = data_T[:,:,m:m+pred_len]
    
    req_time = time.time() - start_time
    print('\nCalculation time: ',req_time)
    
    
    
 ## root-mean-square error & average magnetization calculation...............   
    
    err = np.zeros(pred_len)
    org_up = np.zeros((pred_len,1))
    org_dn = np.zeros((pred_len,1))
    pred_up = np.zeros((pred_len,1))
    pred_dn = np.zeros((pred_len,1))
    for i in range(pred_len):
        err[i] = (np.mean((org[:,:,i]/np.max(org[:,:,i]) - pred[:,:,i]/np.max(pred[:,:,i]))**2))**0.5
        to = org[:,:,i]
        tp = pred[:,:,i]
        org_up[i,0] = np.mean(to[to>0])
        org_dn[i,0] = np.mean(to[to<0])
        pred_up[i,0] = np.mean(tp[tp>0])
        pred_dn[i,0] = np.mean(tp[tp<0])
        
    av_mag = np.hstack((org_up,org_dn,pred_up,pred_dn,err.reshape((-1,1))))
    
    
    
 # printing the calculated RMSE and average magnetization
 #          for actual and predicted data in external files    
   
    # np.savetxt("tdgl_err.txt",err.reshape((-1,1)))
    # np.savetxt("tdgl_av-mag.txt",av_mag)
    
    
    
#............... printing data for few snapshots.........................    
    # temp_org = np.zeros((l*l,2))
    # temp_pred = np.zeros((l*l,2))
    
    # for i in range(l):
    #     for j in range(l):
    #         temp_org[l*i + j,:] = [i,j]
    #         temp_pred[l*i + j,:] = [i,j]
    
    # tp = [100,200,300,400,500,600,700,800,900,1000]
    
    # for i in tp:
    #     temp_org = np.hstack((temp_org,org[:,:,i].reshape((-1,1))))
    #     temp_pred = np.hstack((temp_pred,pred[:,:,i].reshape((-1,1))))
        
    # np.savetxt("sys1_org.txt",temp_org)
    # np.savetxt("sys1_pred.txt",temp_pred)
##########################################################################    
    
    

    
    
#...................Creating movie for the evolution..............................       
    print('\nCreating movie:\n')
    images = []
        
    dif_lim = np.max(np.abs((org[:,:,-1] - pred[:,:,-1])))
    
    NC = 256
    vals = np.ones((NC, 4))
    vals[:, 0] = np.linspace(1, 1, NC)
    vals[:, 1] = np.linspace(1, 0, NC)
    vals[:, 2] = np.linspace(1, 0, NC)
    cmap = ListedColormap(vals)
    cmap1 = ListedColormap(["darkorange", "lightseagreen"])
    
    for i in range(0,pred_len,int(pred_len/100)):
        print(i,end = '\r')
        plt.figure(figsize = [10,8])
        plt.subplot(2,2,1)
        plt.title("Original system")
        plt.pcolor(org[:,:,i].T,cmap=cmap1)
        plt.colorbar()
        plt.clim(np.min(org),np.max(org))
        
        plt.subplot(2,2,2)
        plt.title("Predicted system")
        plt.pcolor(pred[:,:,i].T,cmap=cmap1)
        plt.colorbar()
        plt.clim(np.min(org),np.max(org))
        
        plt.subplot(2,2,3)
        plt.title("Difference")
        plt.pcolor(abs(org[:,:,i] - pred[:,:,i]).T,cmap=cmap)
        plt.colorbar()
        plt.clim(0,dif_lim)
        
        
        plt.subplot(2,2,4)
        plt.title("RMSE")
        plt.plot(err[:i])
        plt.plot(i,err[i],'r+')
        plt.xlim(0,pred_len)
        plt.ylim(np.min(err),np.max(err))
        plt.xlabel("prediction steps")
        plt.ylabel("error")
        
        plt.tight_layout()
        
        plt.savefig('fig_sys1.png')
        plt.close()
        images.append(imageio.imread('fig_sys1.png'))
        
    imageio.mimsave('movie_tdgl.gif', images,duration = 0.1)
