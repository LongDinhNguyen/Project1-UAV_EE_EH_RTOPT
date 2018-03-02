from cvxpy import *
import scipy as sp
import ecos
import matplotlib.pyplot as plt
import time

# sp.random.seed(2030)

# ############################################################
# This code is loaded from Full_code_PA
# ############################################################
# Starting main code for EEmax UAV networks
# ############################################################

bandwidth = 1  # MHz
height = 100  # m
eta = 0.5  # EH efficiency
power_UAV = 5000
power_cir_UAV = 4000
atg_a = 11.95
atg_b = 0.136
noise_variance = sp.multiply(sp.multiply(sp.power(10, sp.divide(-130, 10)), bandwidth), 1e6)
d2d_max = 50

max_chan_realizaion = 1
max_num_d2d_pairs = 10

chan_model = sp.load('chan_model.npz')
max_uav_to_d2d_gains = chan_model['uav']
max_d2d_to_d2d_gains = chan_model['d2d']


# ############################################################
# This loop for a range of num_d2d_pairs
# ############################################################
range_num_d2d_pairs = [10]
# range_num_d2d_pairs = [2, 3, 4, 5, 6, 7, 8, 9, 10]
time_sol_vec_Mon = []
EE_sol_vec_Mon = []

avg = {}
num_infeasible = sp.zeros(len(range_num_d2d_pairs))
for prin in range_num_d2d_pairs:
    num_d2d_pairs = prin
    # rmin = sp.multiply(0.4, sp.log(2))
    time_sol_vec = []
    EE_sol_vec = []

    for Mon in xrange(max_chan_realizaion):
        max_d2d_to_d2d_gains_diff = sp.copy(max_d2d_to_d2d_gains[:, :, Mon])
        sp.fill_diagonal(max_d2d_to_d2d_gains_diff, 0)
        max_d2d_to_d2d_gains_diag = sp.subtract(max_d2d_to_d2d_gains[:, :, Mon], max_d2d_to_d2d_gains_diff)

        uav_to_d2d_gains = max_uav_to_d2d_gains[:num_d2d_pairs, Mon]
        d2d_to_d2d_gains = max_d2d_to_d2d_gains[:num_d2d_pairs, :num_d2d_pairs, Mon]
        d2d_to_d2d_gains_diff = max_d2d_to_d2d_gains_diff[:num_d2d_pairs, :num_d2d_pairs]
        d2d_to_d2d_gains_diag = sp.subtract(d2d_to_d2d_gains, d2d_to_d2d_gains_diff)

        # ############################################################
        # This code is used to find the initial point for EEmax algorithm
        # ############################################################

        theta_ini = Parameter(value=1/(1-0.5))

        phi_n_ini = sp.multiply((theta_ini.value - 1) * eta * sp.divide(power_UAV, 1), uav_to_d2d_gains)
        x_rate = sp.matmul(d2d_to_d2d_gains_diag, phi_n_ini)
        term_rate = sp.matmul(sp.transpose(d2d_to_d2d_gains_diff), phi_n_ini) + 1
        rate_sol_ue = sp.divide(sp.log(sp.add(1, sp.divide(x_rate, term_rate))), theta_ini.value)
        # print rate_sol_ue
        rmin_ref = min(rate_sol_ue)
        print rmin_ref
        if rmin_ref <= 0.4*sp.log(2):
            rmin = rmin_ref
        else:
            rmin = 0.4*sp.log(2)
        # print rmin

        pow_ = NonNegative(num_d2d_pairs)
        objective = Minimize(sum_entries(pow_))

        constraints = []
        constraints.append(d2d_to_d2d_gains_diag * pow_ >= (exp(rmin * theta_ini.value) - 1) * (d2d_to_d2d_gains_diff * pow_ + 1))
        constraints.append(pow_ <= (theta_ini.value - 1) * eta * power_UAV * uav_to_d2d_gains)

        t0 = time.time()

        prob = Problem(objective, constraints)
        prob.solve(solver=ECOS_BB)
        # print prob.status
        # prob.solve(solver=scs)

        term_rate = sp.add(sp.matmul(d2d_to_d2d_gains_diff, pow_.value), 1)
        x_rate = sp.matmul(d2d_to_d2d_gains_diag, pow_.value)
        rate_ini_ue = sp.divide(sp.log(sp.add(1, sp.divide(x_rate, term_rate))), theta_ini.value)
        sum_pow_ini = sp.sum(sp.divide(pow_.value, theta_ini.value)) + sp.subtract(1, sp.divide(1, theta_ini.value)) * eta * power_UAV
        t_ini = sp.add(sum_pow_ini, power_cir_UAV)
        term_phi_ini = sp.divide(sp.sum(rate_ini_ue), t_ini)

        # ############################################################
        # This code is used to solve the EE-max problem
        # ############################################################
        iter = 0
        epsilon = 1
        phi_n_sol = sp.zeros(num_d2d_pairs)
        varphi_sol = 0
        iter_phi = []
        while epsilon >= 1e-2 and iter <= 20:
            iter += 1
            if iter == 1:
               phi_n_ref = pow_.value
               varphi_ref = term_phi_ini
            else:
               phi_n_ref = phi_n_sol
               varphi_ref = varphi_sol

            term_x = sp.matmul(sp.divide(1, d2d_to_d2d_gains_diag, where=d2d_to_d2d_gains_diag != 0), sp.divide(1, phi_n_ref))
            term_y = sp.add(sp.matmul(sp.transpose(d2d_to_d2d_gains_diff), phi_n_ref), 1)

            a_1 = sp.add(sp.log(sp.add(1, sp.divide(1, sp.multiply(term_x, term_y)))),
                         sp.divide(2, sp.add(sp.multiply(term_x, term_y), 1)))
            b_1 = sp.divide(1, sp.multiply(term_x, sp.add(sp.multiply(term_x, term_y), 1)))
            c_1 = sp.divide(1, sp.multiply(term_y, sp.add(sp.multiply(term_x, term_y), 1)))

            phi_n = NonNegative(num_d2d_pairs)

            # obj_1, obj_2, obj_3 = 0, 0, 0
            # for i in xrange(num_d2d_pairs):
            #     obj_1 = obj_1 + a_1[i]
            #     obj_2 = obj_2 + b_1[i]*inv_pos(phi_n[i])/d2d_to_d2d_gains_diag[i, i]
            #     obj_3 = obj_3 + c_1[i]*(d2d_to_d2d_gains_diff[:, i] * phi_n + 1)
            #
            # obj_pow = sum_entries(phi_n)/theta_ini.value
            #
            # obj = (obj_1 - obj_2 - obj_3)/theta_ini.value - varphi_ref * (obj_pow + sp.subtract(1, sp.divide(1, theta_ini.value))*eta*power_UAV + power_cir_UAV)
            # obj_opt = Maximize(obj)

            obj_1 = a_1
            obj_2 = mul_elemwise(sp.reciprocal(d2d_to_d2d_gains_diag, where=d2d_to_d2d_gains_diag != 0) * b_1,
                                 inv_pos(phi_n))
            obj_3 = mul_elemwise(c_1, (sp.transpose(d2d_to_d2d_gains_diff) * phi_n + 1))

            obj_pow = sum_entries(phi_n) / theta_ini.value

            obj = sum_entries(obj_1 - obj_2 - obj_3) / theta_ini.value - \
                  varphi_ref * (
                          obj_pow + sp.subtract(1, sp.divide(1, theta_ini.value)) * eta * power_UAV + power_cir_UAV)
            obj_opt = Maximize(obj)

            # constraints = []
            # constraints.append(phi_n <= (theta_ini.value - 1) * eta * power_UAV * uav_to_d2d_gains)
            # for i in xrange(num_d2d_pairs):
            #     # constraints.append(phi_n[i] <= (theta_ini.value - 1) * eta * power_UAV * uav_to_d2d_gains[i])
            #     constraints.append(a_1[i] - b_1[i]*inv_pos(phi_n[i])/d2d_to_d2d_gains_diag[i, i] -
            #                        c_1[i] * (d2d_to_d2d_gains_diff[:, i] * phi_n + 1) >= theta_ini.value*rmin)

            constraints = []
            constraints.append(phi_n <= (theta_ini.value - 1) * eta * power_UAV * uav_to_d2d_gains)
            constraints.append(
                a_1 -
                mul_elemwise(sp.reciprocal(d2d_to_d2d_gains_diag,
                                           where=d2d_to_d2d_gains_diag != 0) * b_1, inv_pos(phi_n)) -
                mul_elemwise(c_1, (sp.transpose(d2d_to_d2d_gains_diff) * phi_n + 1))
                >= theta_ini.value * rmin)

            t1 = time.time()

            prob = Problem(obj_opt, constraints)
            prob.solve(solver=ECOS_BB)
            # print obj_opt.value

            phi_n_sol = phi_n.value

            x_rate = sp.matmul(d2d_to_d2d_gains_diag, phi_n_sol)
            term_rate = sp.matmul(sp.transpose(d2d_to_d2d_gains_diff), phi_n_sol) + 1
            rate_sol_ue = sp.divide(sp.log(sp.add(1, sp.divide(x_rate, term_rate))), theta_ini.value)

            term_pow_iter = sp.divide(sp.sum(phi_n_sol), theta_ini.value) + sp.subtract(1, sp.divide(1, theta_ini.value))*eta*power_UAV + power_cir_UAV
            varphi_sol = sp.divide(sum(rate_sol_ue), term_pow_iter)

            iter_phi.append(sp.multiply(1e3, sp.divide(varphi_sol, sp.log(2))))

            if iter >= 2:
                epsilon = sp.divide(sp.absolute(sp.subtract(iter_phi[iter - 1], iter_phi[iter - 2])),
                                    sp.absolute(iter_phi[iter - 2]))

        EE_sol_vec.append(sp.multiply(1e3, sp.divide(varphi_sol, sp.log(2))))
        time_sol = (time.time() - t0)
        time_sol_vec.append(time_sol)

        # print 'Number of iterations:', iter
        # print 'Solution:', vars.value

    print rate_sol_ue
    v1 = sp.array(EE_sol_vec)
    EE_sol_vec_Mon.append(sp.mean(v1))
    v2 = sp.array(time_sol_vec)
    time_sol_vec_Mon.append(sp.mean(v2))

print EE_sol_vec_Mon
print time_sol_vec_Mon

sp.savez('result_PA_tau05', EE_PA=EE_sol_vec_Mon, time_PA=time_sol_vec_Mon, x_axis=range_num_d2d_pairs)

# plt.figure(figsize=(8, 6))
# plt.clf()
# plt.plot(range_num_d2d_pairs, time_sol_vec_Mon)
# plt.figure(figsize=(8, 6))
# plt.clf()
# plt.plot(range_num_d2d_pairs, EE_sol_vec_Mon)
# plt.show()
