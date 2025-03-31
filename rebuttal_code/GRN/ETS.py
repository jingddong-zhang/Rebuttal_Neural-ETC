import matplotlib.pyplot as plt
import os.path as osp
from functions import *

setup_seed(10)

def methods(case,mu1,mu2):
    N = 5000  # sample size
    D_in = 2  # input dimension
    H1 = 20  # hidden dimension
    D_out = 1  # output dimension
    data = torch.Tensor(N, D_in).uniform_(-10, 10).requires_grad_(True).to(device)  # -5,5
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
        x_0 = (torch.tensor([[0.62562059, 0.62562059]]) * 10.).requires_grad_(True).to(device)

        i = 0
        max_iters = 600
        learning_rate = 0.05
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
                loss = (L_V + mu_1*torch.sum(data**2,dim=1).view(-1,1) + mu_2*torch.sum(Vx**2,dim=1).view(-1,1)).relu().mean() + 0.1*K_alpha**2


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

        print('K_alpha=',K_alpha.item())
        torch.save(model._lya.state_dict(),osp.join('./data/', case+'_lya mu1={} mu2={}.pkl'.format(mu_1,mu_2)))
        torch.save(model._control.state_dict(),osp.join('./data/', case+'_control mu1={} mu2={}.pkl'.format(mu_1,mu_2)))
        stop = timeit.default_timer()
        ''''''''''''''''''''''''
        # Calculate coefficient in ETC
        ''''''''''''''''''''''''
        a1 = 1.
        a2 = 1.
        b1 = 0.2
        b2 = 0.2
        k = 1.1
        n = 2
        s = 0.5
        K_f = k + a1 * s ** n * 2 * 1.63 / (s ** n + 1.63 ** n) ** 2 + b2 * s ** n * 2 * 1.63 / (
                    s ** n + 1.63 ** n) ** 2
        K_u = (16.3 / 10) ** 2 / (0.5 ** 2 + (16.3 / 10) ** 2)
        inter_time, epi = ETCSDE_cal_coef(mu_1, mu_2, K_u, K_alpha.item(), K_f, 0.0)
        print(inter_time, epi)
        '''
        mu_1=0.9,mu_2=0.9,K_alpha= 3.122598615345626
        '''


        print('\n')
        print("Total time: ", stop - start)

        test_times = torch.linspace(0, 20, 1000)
        s = torch.from_numpy(np.random.choice(np.arange(N, dtype=np.int64), 1))
        init_s = torch.tensor([[0.0582738, 0.85801853]]) * 10.  # +torch.randn(1,2)*0.1
        func = model.GRN
        with torch.no_grad():
            original = odeint(func, init_s, test_times)
            solution = odeint(model.untrigger_fn, init_s, test_times)
        solution = solution.cpu().detach().numpy()
        original = original.cpu().detach().numpy()

        plt.subplot(121)
        plt.plot(test_times, solution[:, 0, 0], label='control')
        plt.plot(test_times, original[:, 0, 0], label='original')
        plt.legend()

        init_state = torch.cat((init_s[0], torch.zeros([2])))
        trajectory_x, trajectory_y, event_times, n_events, traj_events = model.simulate_t(init_state, test_times)
        trajectory_x = trajectory_x.cpu().detach().numpy()
        test_times = test_times.cpu().detach().numpy()

        plt.subplot(122)
        plt.plot(test_times, trajectory_x)
        plt.title('n_events:{}'.format(n_events))

        plt.show()

        out_iters+=1


def table_data(case,mu1,mu2):
    torch.manual_seed(369)
    N = 5000  # sample size
    D_in = 2  # input dimension
    H1 = 20  # hidden dimension
    D_out = 1  # output dimension
    data = torch.Tensor(N, D_in).uniform_(-1.0, 1.0).requires_grad_(True)
    target = (torch.tensor([[0.62562059, 0.62562059]]) * 10.).to(device)
    mu_1 = mu1
    mu_2 = mu2
    ''''''''''''''''''''''''
    # Calculate coefficient in ETC
    ''''''''''''''''''''''''

    a1 = 1.
    a2 = 1.
    b1 = 0.2
    b2 = 0.2
    k = 1.1
    n = 2
    s = 0.5
    K_f = k + a1 * s ** n * 2 * 1.63 / (s ** n + 1.63 ** n) ** 2 + b2 * s ** n * 2 * 1.63 / (s ** n + 1.63 ** n) ** 2
    K_u = (16.3 / 10) ** 2 / (0.5 ** 2 + (16.3 / 10) ** 2)
    inter_time, epi = ETCSDE_cal_coef(mu_1, mu_2, K_u, 3.122598615345626, K_f, 0.0)
    # inter_time, epi = 0.05,0.5
    print(inter_time, epi)

    model = Augment(D_in, H1, D_out, (D_in,), [H1, 1],case,smooth_relu_thresh=0.1, eps=1e-3,ETCSDE_tau=inter_time,ETCSDE_epi=epi).to(device)
    model._control.load_state_dict(torch.load(osp.join('./data/', case + '_control mu1={} mu2={}.pkl'.format(mu_1,mu_2))))
    model._lya.load_state_dict(torch.load(osp.join('./data/', case + '_lya mu1={} mu2={}.pkl'.format(mu_1,mu_2))))
    test_times = torch.linspace(0, 20, 1000).to(device)
    start = timeit.default_timer()
    init_s = torch.tensor([[0.0582738, 0.85801853]]) * 10.
    init_state = torch.cat((init_s[0], torch.zeros([2])))

    event_num = []
    min_traj = []
    min_inter = []
    min_traj_events = []
    var_list = []
    seed_list = [2, 4, 5, 6, 7]
    for i in range(5):
        with torch.no_grad():
            seed = seed_list[i]
            # seed = i
            np.random.seed(seed)
            s = torch.from_numpy(np.random.choice(np.arange(N, dtype=np.int64), 1))
            init_noise = torch.cat((data[s][0, 0:2], torch.zeros([2]).to(device))).to(device)
            trajectory_x, trajectory_y, event_times, n_events, traj_events = model.simulate_t(init_state + init_noise,
                                                                                              test_times)
            traj_events += -target
            event_times = torch.tensor(event_times)
            event_num.append(n_events)
            min_traj.append(
                torch.min(torch.sqrt((trajectory_x - target[0, 0]) ** 2 + (trajectory_y - target[0, 0]) ** 2)))
            cat_data = torch.cat((trajectory_x.unsqueeze(1), trajectory_y.unsqueeze(1)), dim=1)
            var_list.append(variance(cat_data, target, n=900))
            min_inter.append((event_times[1:] - event_times[:-1]).min())
            if len(traj_events) >= 11:
                min_traj_events.append(torch.linalg.norm(traj_events[10], ord=2))
            else:
                min_traj_events.append(torch.linalg.norm(traj_events[-1], ord=2))
            print(seed, trajectory_x[-1], min_traj[i], n_events, min_inter[i])
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
average inference time=1.66935718
29.0 tensor(0.0871) tensor(0.0466) tensor(1.4459) tensor(0.2289)
8.17312669668102 tensor(0.1259) tensor(3.2546e-17) tensor(0.7010)

modified parameters: inter_time=0.05, epi=0.5
average inference time=2.82834524
40.4 tensor(0.1682) tensor(0.1028) tensor(7.4219) tensor(25.0244)
14.263239463740346 tensor(0.1232) tensor(0.0239) tensor(4.6910)
'''
