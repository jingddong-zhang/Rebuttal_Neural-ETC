import matplotlib.pyplot as plt
import os.path as osp
from functions import *

setup_seed(10)

'''
PGD based counterexamples tool
'''
def pgd_attack_boundary(
    x0, f, eps, steps=10, boundary=5.0,lower_boundary=None, upper_boundary=None, direction="maximize"
):
    """
    Use adversarial attack (PGD) to find violating points.
    Args:
      x0: initialization points, in [batch, state_dim].
      f: function f(x) to find the worst case x to maximize.
      eps: perturbation added to x0.
      steps: number of pgd steps.
      lower_boundary: absolute lower bounds of x.
      upper_boundary: absolute upper bounds of x.
    """
    # Set all parameters without gradient, this can speedup things significantly
    grad_status = {}
    try:
        for p in f.parameters():
            grad_status[p] = p.requires_grad
            p.requires_grad_(False)
    except:
        pass

    step_size = eps / steps * 2
    noise = torch.randn_like(x0) * step_size
    if lower_boundary is not None:
        lower_boundary = torch.max(lower_boundary, x0 - eps)
    else:
        lower_boundary = x0 - eps
    if upper_boundary is not None:
        upper_boundary = torch.min(upper_boundary, x0 + eps)
    else:
        upper_boundary = x0 + eps
    x = x0.detach().clone().requires_grad_()
    # Save the best x and best loss.
    best_x = torch.clone(x).detach().requires_grad_(False)
    fill_value = float("-inf") if direction == "maximize" else float("inf")
    best_loss = torch.full(
        size=(x.size(0),),
        requires_grad=False,
        fill_value=fill_value,
        device=x.device,
        dtype=x.dtype,
    )
    for i in range(steps):
        output = f(x).squeeze(1)
        # output = torch.clamp(f(x).squeeze(1), max=0)
        output.mean().backward()
        if direction == "maximize":
            improved_mask = output >= best_loss
        else:
            improved_mask = output <= best_loss
        best_x[improved_mask] = x[improved_mask]
        best_loss[improved_mask] = output[improved_mask]
        # print(f'step = {i}', output.view(-1).detach())
        # print(x.detach(), best_x)
        noise = torch.randn_like(x0) * step_size / (i + 1)
        if direction == "maximize":
            # x = (
            #     (
            #         torch.clamp(
            #             x + torch.sign(x.grad) * step_size + noise,
            #             min=lower_boundary,
            #             max=upper_boundary,
            #         )
            #     )
            #     .detach()
            #     .requires_grad_()
            # )

            # print(x.shape)
            c_x = x + torch.sign(x.grad) * step_size + noise
            min_abs, ind_x = torch.min(torch.abs(c_x), dim=1, keepdim=True)
            improved_mask = torch.abs(c_x) == min_abs
            normal_c_x = c_x / torch.abs(c_x) * boundary
            normal_c_x[improved_mask] = c_x[improved_mask]
            x = normal_c_x.detach().requires_grad_()
        else:
            x = (
                (
                    torch.clamp(
                        x - torch.sign(x.grad) * step_size + noise,
                        min=lower_boundary,
                        max=upper_boundary,
                    )
                )
                .detach()
                .requires_grad_()
            )

    # restore the gradient requirement for model parameters
    try:
        for p in f.parameters():
            p.requires_grad_(grad_status[p])
    except:
        pass
    return best_x


def methods(case):
    N = 5000  # sample size
    D_in = 2  # input dimension
    H1 = 20  # hidden dimension
    D_out = 1  # output dimension
    # data = torch.Tensor(N, D_in).uniform_(-5, 5).requires_grad_(True).to(device)  # -5,5
    lo,up = -12,12 #concerned region, attractor of lorenz
    N_boundary = 100 # boundary sample size
    boundary = 10.0 # region [-boundary,boundary]^D_in
    out_iters = 0
    N1 = 600
    kappa = 0.1
    dt = 0.01
    c0 = 0.1
    c1 = 0.1
    c2 = 0.001
    gamma = 1.1
    import scipy.io as scio
    lqr_data = scio.loadmat('./data/lqr_data.mat')
    S = torch.from_numpy(np.array(lqr_data['S1'])).to(device)
    data_candidata = torch.Tensor(50, D_in).uniform_(-10, 10)
    data_candidata *= torch.sum(data_candidata*torch.mm(S,data_candidata.T).T,dim=1).view(-1,1)
    data_candidata = data_candidata.requires_grad_(True).to(device)
    while out_iters < 1:
        # break
        start = timeit.default_timer()
        model = Augment(D_in, H1, D_out, (D_in,), [H1, 1],case).to(device)
        x_0 = (torch.tensor([[0.62562059, 0.62562059]]) * 10.).requires_grad_(True).to(device)
        i = 0
        max_iters = N1
        learning_rate = 0.1
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        L = []

        def func_V(data):
            V = model._lya(data)
            return -V


        while i < max_iters:
            # break
            '''
            sample boundary data and calculate rho
            '''
            init_data = torch.Tensor(N_boundary, D_in).uniform_(-10, 10)
            min_abs, ind_x = torch.min(torch.abs(init_data), dim=1, keepdim=True)
            improved_mask = torch.abs(init_data) == min_abs
            boundary_data = init_data / torch.abs(init_data) * boundary
            boundary_data[improved_mask] = init_data[improved_mask]
            pgd_boundary_data = pgd_attack_boundary(boundary_data, func_V, 1.0, 5,boundary).requires_grad_(True).to(device)
            rho = gamma*torch.min(model._lya(pgd_boundary_data)).item()

            '''
            sample counterexamples and train 
            '''
            def func_L_dV(data):
                # print('current rho=',rho)
                V = model._lya(data)
                Vx = torch.autograd.grad(V.sum(), data, create_graph=True)[0]
                f_u =  model.untrigger_fn(1.0, data)
                L_V = (Vx * f_u).sum(dim=1).view(-1, 1)
                next_data = data + dt * f_u
                H = (next_data - up).relu().sum(dim=1).view(-1, 1) + (lo - next_data).relu().sum(dim=1).view(-1, 1)
                cat_ = torch.cat(((L_V + kappa * V).relu() + c0 * H, rho - V), dim=1)
                L_dV, _ = torch.min(cat_, dim=1, keepdim=True)
                # L_dV = (L_V + kappa * V).relu() + c0 * H
                L_dV = L_dV.relu()
                return L_dV

            iter_data = torch.Tensor(N, D_in).uniform_(-10, 10) # -5,5
            data = pgd_attack(iter_data,func_L_dV,1.0,steps=10,lower_boundary=torch.tensor([lo]),upper_boundary=torch.tensor([up])).requires_grad_(True).to(device)
            V = model._lya(data)
            Vx = torch.autograd.grad(V.sum(), data, create_graph=True)[0]
            f_u = model.untrigger_fn(1.0, data)
            L_V = (Vx * f_u).sum(dim=1).view(-1, 1)
            next_data = data+dt*f_u
            H = (next_data-up).relu().sum(dim=1).view(-1,1)+(lo-next_data).relu().sum(dim=1).view(-1,1)
            cat_ = torch.cat(((L_V + kappa * V).relu() + c0 * H, rho - V), dim=1)
            L_dV, _ = torch.min(cat_, dim=1, keepdim=True)
            # L_dV = (L_V + kappa * V).relu() + c0 * H
            L_dV = L_dV.relu().sum()
            L_roa = (model._lya(data_candidata)/rho-1).relu().sum()
            L1 = 0
            for param in model.parameters():
                L1 += torch.sum(torch.abs(param))
            loss = L_dV + c2 * L1+ c1 * L_roa

            L.append(loss)
            print(i, 'total loss=',loss.item(),'L_dV=',L_dV.item()) #,'L_roa=',L_roa.item(),'rho=',rho

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if i > 300 and loss<=1e-4:
            #     break

            # stop = timeit.default_timer()
            # print('per:',stop-start)
            i += 1
        # print(q)
        torch.save(model._lya.state_dict(),osp.join('./data/', case+'_lya.pkl'))
        torch.save(model._control.state_dict(),osp.join('./data/', case+'_control.pkl'))
        stop = timeit.default_timer()


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

# methods('PGDNLC_quad')
# methods('PGDNLC')

def table_data(case):
    torch.manual_seed(369)
    N = 5000  # sample size
    D_in = 2  # input dimension
    H1 = 20  # hidden dimension
    D_out = 1  # output dimension
    data = torch.Tensor(N, D_in).uniform_(-1.0, 1.0).requires_grad_(True)
    target = (torch.tensor([[0.62562059, 0.62562059]]) * 10.).to(device)

    model = Augment(D_in, H1, D_out, (D_in,), [H1, 1],case).to(device)
    model._control.load_state_dict(torch.load(osp.join('./data/', case + '_control.pkl')))
    model._lya.load_state_dict(torch.load(osp.join('./data/', case + '_lya.pkl')))
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

table_data('PGDNLC')
'''
PGDNLC
average inference time=1.22366226
22.8 tensor(0.0661) tensor(0.3788) tensor(0.3948) tensor(0.0079)
2.4 tensor(0.0366) tensor(0.1500) tensor(0.0084)
'''





