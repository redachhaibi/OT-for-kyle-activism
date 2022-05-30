import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def solve_sinkhorn(M, N, m, epsilon, params, debug=False):
    # Define grids for variables y,z,xi
    grid_y  = np.linspace(-m,m,N)
    grid_z  = np.linspace(-m,m,N)
    grid_xi = np.linspace(0,1,M)
    [Y,Z,Xi] = np.meshgrid(grid_y,grid_z,grid_xi)
    dimensions = (N, N, M) 
    
    # Increments
    dy  = 2*m/N
    dz  = dy
    dxi = 1.0/M

    # Copy hyperparameters
    sigma = params['sigma']
    T     = params['T']
    sigma_beta = params['sigma_beta']
    m_beta     = params['m_beta']
    delta      = params['delta']
    x_star     = params['x_star']
    
    # Target measures
    # - Law of Y
    v_eff = sigma*sigma*T + sigma_beta*sigma_beta
    a = dy*np.exp( -0.5*grid_y*grid_y/v_eff)/np.sqrt(2*np.pi*v_eff)
    # - Law of (Z-beta, Xi)
    v_eff = sigma*sigma*T + sigma_beta*sigma_beta
    b = dz*np.exp( -m_beta*grid_y - 0.5*grid_y*grid_y/v_eff)/np.sqrt(2*np.pi*v_eff)
    [b1, b2] = np.meshgrid(b, dxi* np.ones_like(grid_xi))
    b = b1*b2 # Product measure
    b = b.T   # For some reason, dim order is correct this way
    a = a/np.sum(a)
    b = b/np.sum(b)
    if debug:
        print( "Grid for measure a:", a.shape )
        print( "Grid for measure b:", b.shape )

    # Kernel
    surplus = (Y-Z)*Xi + delta*np.fmax(Y-Z-x_star, 0)
    K1 = np.exp( surplus/epsilon )
    if debug:
        print( "Grid for kernel:", K1.shape )

    #
    # Launch Sinkhorn
    v = np.ones_like(b)
    K  = lambda x : np.tensordot( K1, x, axes=[[1,2], [0,1]])
    KT = lambda x : np.tensordot( K1, x, axes=[[0], [0]])

    count = 1000
    errors = []
    for i in range(count):
        u = a / K(v)
        v = b / KT(u)
        e = np.linalg.norm( u[None,:]*K(v) - a )
        errors.append( e )
    #
    # Plot
    errors_sinkhorn = errors
    plt.yscale('log')
    plt.plot(errors)
    plt.ylabel('Error in log-scale')
    plt.xlabel('Number of iterations')
    #
    return u, K1, v

def plot_level_set(plan, M, N, m):
    print("Total mass: ", np.sum(plan))
    # Define grids for variables y,z,xi
    grid_y  = np.linspace(-m,m,N)
    grid_z  = np.linspace(-m,m,N)
    grid_xi = np.linspace(0,1,M)
    # Set colors
    plt.figure(figsize=(10, 8), dpi=80)
    cmap = plt.get_cmap('viridis',N)
    norm = matplotlib.colors.Normalize(vmin=0,vmax=1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    #
    xi_old  = np.zeros_like( grid_z )
    for i in range(N):
        # Slice for fixed y
        # Note: Slice = Conditionnal distribution (Z,Xi | Y)
        plan_slice = plan[i,:,:]
        plan_slice = plan_slice/np.sum(plan_slice)
        # Compute mean of Xi conditionally to Z
        xi = np.dot( plan_slice, grid_xi )/np.sum(plan_slice, axis=1)
        plt.fill_between(x=grid_z, y1=xi_old, y2=xi, color=cmap( norm(i/N) ) )
        xi_old = xi
    plt.xlabel('Z axis')
    plt.ylabel('Xi axis')
    plt.colorbar(sm, ticks=np.linspace(-3,3,21), 
                 boundaries=np.arange(-3,3,0.1))
    plt.show()

def plot_level_set2(plan, M, N, m):
    print("Total mass: ", np.sum(plan))
    # Define grids for variables y,z,xi
    grid_y  = np.linspace(-m,m,N)
    grid_z  = np.linspace(-m,m,N)
    grid_xi = np.linspace(0,1,M)
    # Set colors
    plt.figure(figsize=(10, 8), dpi=80)
    cmap = plt.get_cmap('viridis',N)
    norm = matplotlib.colors.Normalize(vmin=0,vmax=1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    #
    xi_old  = np.zeros_like( grid_z )
    for i in range(N):
        # Slice for fixed y
        # Note: Slice = Conditionnal distribution (Z,Xi | Y)
        plan_slice = plan[i,:,:]
        plan_slice = plan_slice/np.sum(plan_slice)
        # Compute mean of Xi conditionally to Z
        z = grid_z
        xi = np.dot( plan_slice, grid_xi )/np.sum(plan_slice, axis=1)
        plt.plot( grid_z, xi )
    plt.xlabel('Z axis')
    plt.ylabel('Xi axis')
    plt.show()

def compute_I(plan, M, N, m, debug=False):
    if debug:
        print("Total mass: ", np.sum(plan))
    # Define grids for variables y,z,xi
    grid_y  = np.linspace(-m,m,N)
    grid_z  = np.linspace(-m,m,N)
    grid_xi = np.linspace(0,1,M)

    # Height is conditional mean Y | (Z,Xi)
    conditional_mass = np.tensordot( np.ones_like(grid_y), plan, axes=[ [0], [0]] )
    height = plan/conditional_mass[None, :,:]
    height = np.tensordot( grid_y, height, axes=[ [0], [0]] )
    
    return grid_z, grid_xi, height.T