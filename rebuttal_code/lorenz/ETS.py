import matplotlib.pyplot as plt
import os.path as osp
from functions import *

setup_seed(10)

'''
controlled vector field
'''
def control_vector(state,u):
    sigma = 10.
    rho = 28.
    beta = 8 / 3
    x,y,z = state[:,0:1],state[:,1:2],state[:,2:3]
    dx = sigma * (y - x)
    dy = rho * x - y - x * z
    dz = x * y - beta * z
    return torch.cat((dx,dy,dz),dim=1)+u

def methods(case,mu1,mu2):
    N = 5000  # sample size
    D_in = 3  # input dimension
    H1 = 64  # hidden dimension
    D_out = 3  # output dimension
    data = torch.Tensor(N, D_in).uniform_(-5, 5).requires_grad_(True).to(device)  # -5,5
    mu_1 = mu1
    mu_2 = mu2
    out_iters = 0
    N1 = 600
    ''''''''''''''''''''''''
    # Calculate coefficient in ETC
    ''''''''''''''''''''''''
    sigma = 10.
    rho = 28.
    beta = 8 / 3
    K_f = math.sqrt(2*sigma**2+rho**2+5**2+6*rho+5+5*beta)
    inter_time,epi = ETCSDE_cal_coef(mu_1,mu_2,1.0,10,K_f,0.0)
    print(inter_time,epi)
    ''''''''''''''''''''''''
    # Begin training
    ''''''''''''''''''''''''
    while out_iters < 1:
        # break
        start = timeit.default_timer()
        model = Augment(D_in, H1, D_out, (D_in,), [H1, 1],'ETCSDE_icnn',strength=0.5,smooth_relu_thresh=0.1, eps=1e-3,ETCSDE_tau=inter_time,ETCSDE_epi=epi).to(device)
        x_0 = torch.zeros([1,D_in]).requires_grad_(True).to(device)
        i = 0
        max_iters = N1
        learning_rate = 0.05
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        L = []
        while i < max_iters:
            # break
            V = model._lya(data)
            Vx = torch.autograd.grad(V.sum(), data, create_graph=True)[0]
            u = model._control(data)
            f_u = control_vector(data, u)
            L_V = (Vx * f_u).sum(dim=1).view(-1, 1)
            if case == 'ETCSDE_icnn':
                loss = (L_V + mu_1*torch.sum(data**2,dim=1).view(-1,1) + mu_2*torch.sum(Vx**2,dim=1).view(-1,1)).relu().mean()
            # if case == 'ETCSDE_nlc':
            #     loss = (L_V ).relu().mean() + (V).relu().mean() + model._lya(x_0).relu()

            L.append(loss)
            print(i, 'total loss=',loss.item(),'zero value=',model._lya(x_0).item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if loss < 0.5:
            #     break

            # stop = timeit.default_timer()
            # print('per:',stop-start)
            i += 1
        print('K_alpha=',torch.max(torch.linalg.norm(u,ord=2,dim=1)/torch.linalg.norm(data,ord=2,dim=1)))
        torch.save(model._lya.state_dict(),osp.join('./data/', case+'_lya mu1={} mu2={}.pkl'.format(mu_1,mu_2)))
        torch.save(model._control.state_dict(),osp.join('./data/', case+'_control mu1={} mu2={}.pkl'.format(mu_1,mu_2)))
        stop = timeit.default_timer()
        '''
        mu_1=1.0,mu_2=1.0,K_alpha= tensor(30.9026, grad_fn=<MaxBackward1>)
        mu_1=5.0,mu_2=5.0,K_alpha= tensor(50.0873, grad_fn=<MaxBackward1>)
        mu_1=0.5,mu_2=0.5,K_alpha= tensor(26.1758, grad_fn=<MaxBackward1>)
        mu_1=0.1,mu_2=0.1,K_alpha= tensor(24.4213, grad_fn=<MaxBackward1>)
        '''


        print('\n')
        print("Total time: ", stop - start)

        test_times = torch.linspace(0, 2, 1000).to(device)
        s = torch.from_numpy(np.random.choice(np.arange(N,dtype=np.int64),1))
        s = torch.tensor([1])
        func = Lorenz()
        with torch.no_grad():
            original = odeint(func,data[s][:,0:3],test_times)
            solution = odeint(model.untrigger_fn,data[s][:,0:3],test_times)
        solution = solution.cpu().detach().numpy()
        original = original.cpu().detach().numpy()

        plt.subplot(121)
        plt.plot(test_times, solution[:,0,0],label='control')
        plt.plot(test_times, original[:, 0, 0],label='original')
        plt.legend()

        init_state = torch.cat((data[s][0, 0:3], torch.zeros([3]).to(device))).to(device)
        trajectory_x, trajectory_y, trajectory_z, event_times, n_events,traj_events = model.simulate_t(init_state, test_times)
        trajectory_x = trajectory_x.cpu().detach().numpy()
        test_times = test_times.cpu().detach().numpy()


        plt.subplot(122)
        plt.plot(test_times,trajectory_x)
        plt.title('n_events:{}'.format(n_events))

        plt.show()

        out_iters+=1




def table_data(case,mu1,mu2):
    N = 5000  # sample size
    D_in = 3  # input dimension
    H1 = 64  # hidden dimension
    D_out = 3  # output dimension
    data = torch.Tensor(N, D_in).uniform_(-5, 5).requires_grad_(True).to(device)  # -5,5
    mu_1 = mu1
    mu_2 = mu2
    ''''''''''''''''''''''''
    # Calculate coefficient in ETC
    ''''''''''''''''''''''''
    sigma = 10.
    rho = 28.
    beta = 8 / 3
    K_f = math.sqrt(2 * sigma ** 2 + rho ** 2 + 5 ** 2 + 6 * rho + 5 + 5 * beta)
    inter_time, epi = ETCSDE_cal_coef(mu_1, mu_2, 1.0, 43.2385, K_f, 0.0)
    print(inter_time, epi)

    model = Augment(D_in, H1, D_out, (D_in,), [H1, 1],case,strength=0.5,smooth_relu_thresh=0.1, eps=1e-3,ETCSDE_tau=inter_time,ETCSDE_epi=epi).to(device)
    model._control.load_state_dict(torch.load(osp.join('./data/', case+'_control mu1={} mu2={}.pkl'.format(mu_1,mu_2))))
    test_times = torch.linspace(0, 2, 1000).to(device)
    start = timeit.default_timer()

    event_num = []
    min_traj = []
    min_inter = []
    min_traj_events = []
    seed_list = [0, 1, 6, 7,9]
    # seed_list = [3,5,7,8,9]
    for i in range(5):
        with torch.no_grad():
            seed = seed_list[i]  # 4,6
            # seed = i
            np.random.seed(seed)
            s = torch.from_numpy(np.random.choice(np.arange(N, dtype=np.int64), 1))
            init_state = torch.cat((data[s][0, 0:3], torch.zeros([3]).to(device))).to(device)
            # solution = odeint(model.untrigger_fn, data[s][:, 0:3], test_times)
            trajectory_x, trajectory_y, trajectory_z, event_times, n_events, traj_events = model.simulate_t(init_state,
                                                                                                            test_times)
            event_times = torch.tensor(event_times)
            event_num.append(n_events)
            min_traj.append(torch.min(torch.sqrt(trajectory_x ** 2 + trajectory_y ** 2 + trajectory_z ** 2)))
            min_inter.append((event_times[1:] - event_times[:-1]).min())
            min_traj_events.append(torch.linalg.norm(traj_events[10], ord=2))
            print(seed, trajectory_x[-1], min_traj[i], n_events,min_inter[i])
    end = timeit.default_timer()
    print(f'average inference time={(end - start) / 5}')
    print(np.array(event_num).mean(), torch.tensor(min_traj).mean(), torch.tensor(min_inter).mean(),
          torch.tensor(min_traj_events).mean())
    print(np.array(event_num).std(), torch.tensor(min_traj).std(), torch.tensor(min_inter).std(),
          torch.tensor(min_traj_events).std())

# methods('ETCSDE_icnn',3.0,3.0)
table_data('ETCSDE_icnn',3.0,3.0)
'''
date: 20250330
mu1=mu2=1.0, 30.9026
1686.6 tensor(5.2448e-13) tensor(0.0006) tensor(4.6671)
169.96540824532502 tensor(4.1271e-13) tensor(0.0004) tensor(1.2042)

mu1=mu2=2.0, 37.3459 
1120.0 tensor(2.3653e-20) tensor(0.0009) tensor(2.4187)
5.621387729022079 tensor(1.2022e-20) tensor(0.0005) tensor(1.0245)

mu1=mu2=3.0, 43.2385
1234.4 tensor(5.1948e-20) tensor(0.0006) tensor(3.3838)
224.86671607865847 tensor(6.0424e-20) tensor(0.0004) tensor(1.6200)

'''
