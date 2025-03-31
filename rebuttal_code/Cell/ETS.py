import matplotlib.pyplot as plt
import os.path as osp
from functions import *

setup_seed(10)

def methods(case,mu1,mu2):
    N = 5000  # sample size
    D_in = 100  # input dimension
    H1 = 2 * D_in  # hidden dimension
    D_out = 100  # output dimension
    data = torch.Tensor(N, D_in).uniform_(-10, 10).requires_grad_(True).to(device)  # -5,5
    x_0 = torch.from_numpy(np.load('./data/target_100.npy')).view([1, -1]).requires_grad_(True).to(device)
    mu_1 = mu1
    mu_2 = mu2
    out_iters = 0


    ''''''''''''''''''''''''
    # Begin training
    ''''''''''''''''''''''''
    while out_iters < 1:
        # break
        start = timeit.default_timer()
        model = Augment(D_in, H1, D_out, (D_in,), [H1, 1], case).to(device)
        i = 0
        max_iters = 600
        learning_rate = 0.01
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        L = []
        while i < max_iters:
            # break
            V = model._lya(data)
            Vx = torch.autograd.grad(V.sum(), data, create_graph=True)[0]
            u = model._control(data)
            f_u = model.untrigger_fn(1.0, data)
            L_V = (Vx * f_u).sum(dim=1).view(-1, 1)
            K_alpha = torch.max(torch.linalg.norm(u, ord=2, dim=1) / torch.linalg.norm(data - x_0, ord=2, dim=1))
            if case == 'ETCSDE_icnn':
                loss = (L_V + mu_1*torch.sum(data**2,dim=1).view(-1,1) + mu_2*torch.sum(Vx**2,dim=1).view(-1,1)).relu().mean() #+ 0.1*K_alpha**2


            L.append(loss)
            print(i, 'total loss=',loss.item(),'zero value=',model._lya(x_0).item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            i += 1

        print('K_alpha=',K_alpha.item())
        torch.save(model._lya.state_dict(),osp.join('./data/', case+'_lya mu1={} mu2={} reg_free.pkl'.format(mu_1,mu_2)))
        torch.save(model._control.state_dict(),osp.join('./data/', case+'_control mu1={} mu2={}.pkl reg_free'.format(mu_1,mu_2)))
        stop = timeit.default_timer()

        '''
        mu_1=0.9,mu_2=0.9,K_alpha= 3.422224084607538 or reg_free 10.232230515679893
        '''


        print('\n')
        print("Total time: ", stop - start)

        test_times = torch.linspace(0, 30, 1000)
        s = torch.from_numpy(np.random.choice(np.arange(N, dtype=np.int64), 1))
        init_s = torch.zeros([D_in]).view(-1, D_in) + torch.from_numpy(np.random.uniform(-0.5, 0.5, [1, D_in]))
        func = model.Cell
        with torch.no_grad():
            original = odeint(func, init_s, test_times)
            solution = odeint(model.untrigger_fn, init_s, test_times)
        solution = solution.cpu().detach().numpy()
        original = original.cpu().detach().numpy()

        plt.subplot(121)
        plt.plot(test_times, solution[:, 0, 0], label='control')
        plt.plot(test_times, original[:, 0, 0], label='original')
        plt.legend()

        init_state = torch.zeros([2 * D_in])
        init_state[0:D_in] += torch.from_numpy(np.random.uniform(-0.5, 0.5, [D_in]))
        trajectory, event_times, n_events, traj_events = model.simulate_t(init_state, test_times)
        trajectory = trajectory.cpu().detach().numpy()
        test_times = test_times.cpu().detach().numpy()

        plt.subplot(122)
        plt.plot(test_times, trajectory[:, 0])
        plt.title('n_events:{}'.format(n_events))

        plt.show()

        out_iters+=1


def table_data(case,mu1,mu2):
    torch.manual_seed(369)
    N = 1000  # sample size
    D_in = 100  # input dimension
    H1 = 2 * D_in  # hidden dimension
    D_out = 100  # output dimension
    data = torch.Tensor(N, D_in).uniform_(-10, 10).requires_grad_(True)
    target = torch.from_numpy(np.load('./data/target_100.npy')).view([1, -1])
    mu_1 = mu1
    mu_2 = mu2
    ''''''''''''''''''''''''
    # Calculate coefficient in ETC
    ''''''''''''''''''''''''

    B = 1.
    A = torch.from_numpy(np.load('./data/A_100.npy'))
    K_f = 10*(B+torch.max(A)*2*20/(1+20**2)**2)
    K_u = (20) ** 2 / (1.0 ** 2 + 20 ** 2)
    inter_time, epi = ETCSDE_cal_coef(mu_1, mu_2, K_u, 3.422224084607538, K_f, 0.0)
    # inter_time, epi = 0.05,0.5
    print(inter_time, epi)

    model = Augment(D_in, H1, D_out, (D_in,), [H1, 1],case,dim=100,smooth_relu_thresh=0.1, eps=1e-3,ETCSDE_tau=inter_time,ETCSDE_epi=epi).to(device)
    model._control.load_state_dict(torch.load(osp.join('./data/', case + '_control mu1={} mu2={}.pkl'.format(mu_1,mu_2))))
    model._lya.load_state_dict(torch.load(osp.join('./data/', case + '_lya mu1={} mu2={}.pkl'.format(mu_1,mu_2))))
    test_times = torch.linspace(0, 30, 1000).to(device)
    start = timeit.default_timer()
    init_state = torch.zeros([2 * D_in])

    event_num = []
    min_traj = []
    min_inter = []
    min_traj_events = []
    var_list = []
    seed_list = [2, 4, 5, 6, 7]
    for i in range(5):
        with torch.no_grad():
            # seed = seed_list[i]
            seed = i
            np.random.seed(seed)
            init_state[0:D_in] += torch.from_numpy(np.random.uniform(-0.5, 0.5, [D_in]))
            trajectory, event_times, n_events, traj_events = model.simulate_t(init_state, test_times)
            traj_events += -target
            trajectory += -target
            event_times = torch.tensor(event_times)
            event_num.append(n_events)
            min_traj.append(
                torch.min(torch.sqrt(torch.sum(trajectory ** 2, dim=1))))
            var_list.append(variance(trajectory, torch.zeros_like(trajectory), n=900))
            min_inter.append((event_times[1:] - event_times[:-1]).min())
            # min_inter.append(0.0)

            if len(traj_events) >= 11:
                min_traj_events.append(torch.linalg.norm(traj_events[10], ord=2))
            else:
                min_traj_events.append(torch.linalg.norm(traj_events[-1], ord=2))
            print(seed, min_traj[i], n_events, min_inter[i])
    end = timeit.default_timer()
    print(f'average inference time={(end - start) / 5}')
    print(np.array(event_num).mean(), torch.tensor(min_traj).mean(), torch.tensor(min_inter).mean(),
          torch.tensor(min_traj_events).mean(), torch.tensor(var_list).mean())
    print(np.array(event_num).std(), torch.tensor(min_traj).std(), torch.tensor(min_inter).std(),
          torch.tensor(min_traj_events).std())

# methods('ETCSDE_icnn',0.9,0.9)
table_data('ETCSDE_icnn',0.9,0.9)

'''
mu1=mu2=0.9, 3.122598615345626
average inference time=4.33970698
11.4 tensor(0.4574) tensor(0.0269) tensor(4.9613) tensor(0.3142)
1.0198039027185568 tensor(0.3507) tensor(0.0017) tensor(7.5251)
'''
