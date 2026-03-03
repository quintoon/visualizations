import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


global sigma_proposal, times, y, y_err

# choose mcmc hyperparameter
sigma_proposal = 2

# generate some fake data
times = np.linspace(0, 10, 50000)
m_true = (np.random.randint(1,10),np.random.randint(1,10),np.random.randint(1,10))
y = m_true[0] * times**2 + m_true[1]*times + m_true[2] 
y_err = 25


def calculate_posterior_prob(m, times=times, y=y, y_err=y_err):
    a, b, c = m
    model_prediction = a * (times ** 2) + (b * times) + c
    residual = model_prediction - y
    chi2_array = (residual / y_err) ** 2
    chi2 = np.sum(chi2_array)
    posterior_prob = np.exp(-0.5 * chi2)

    posterior_prob_list = []
    posterior_prob_list.append(posterior_prob)
    print(f"{posterior_prob_list}")
    list_m = []
    list_m.append(m)
    print(f"{m}")

    return posterior_prob


def pick_next_state(m_current):
    m_new = np.random.normal(loc=m_current, scale=sigma_proposal)
    return m_new


# def calculate_proposal_prob(m_new, m_current):
#     proposal_prob = norm.pdf(m_new, loc=m_current, scale=sigma_proposal)
#     return proposal_prob


def calculate_acceptance_prob(m_current, m_new):

    acceptance_prob = np.min(
        [
            1,
            calculate_posterior_prob(m_new) / calculate_posterior_prob(m_current),
            # * calculate_proposal_prob(m_current, m_new)
            # / calculate_proposal_prob(m_new, m_current),
        ]
    )

    return acceptance_prob


def mh_mcmc(m_initial_guess, num_steps):

    markov_chain = []

    m_current = m_initial_guess
    for i in range(num_steps):

        m_proposal = pick_next_state(m_current)
        acceptance_prob = calculate_acceptance_prob(m_current, m_proposal)

        random_number = np.random.uniform(0, 1, size=1)
        if acceptance_prob >= random_number:
            m_current = m_proposal

        markov_chain.append(m_current)

    return np.array(markov_chain)

if __name__ == "__main__":

    np.random.seed(10)

    m_initial_guess = (1,1,1)
    num_steps = 50000

    markov_chain = mh_mcmc(m_initial_guess=m_initial_guess, num_steps=num_steps)
    a_best = np.mean(markov_chain[:,0])
    b_best = np.mean(markov_chain[:,1])
    c_best = np.mean(markov_chain[:,2])
    y_best = a_best *times**2 + b_best * times + c_best

    plt.plot(times, y_best, label='best fit')
    plt.plot(times, y, label='original')
    plt.savefig("mh_mcmc1.png",dpi=250)
    plt.legend()
    plt.show()

    plt.plot(times, markov_chain[:,0], label='a mcmc searcher')
    plt.legend()
    plt.savefig("a_walker.png",dpi=250)
    plt.show()
    plt.plot(times, markov_chain[:,1], label='b mcmc searcher')
    plt.legend()
    plt.savefig("b_walker.png",dpi=250)
    plt.show()
    plt.plot(times, markov_chain[:,2], label='c mcmc searcher')
    plt.legend()
    plt.savefig("c_walker.png",dpi=250)
    plt.show()
