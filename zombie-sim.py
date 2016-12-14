import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def sim(population=10000, infected=0.025, exp=(1, 1, 1, 1, 1, 1, 1, 1, 1, 1), z_ratio=1, exp_growth=False):
    # wartości doświadczenia
    types = np.arange(0, 100, 10, dtype=int)

    # względne wielkości poszczególnych grup
    freq = np.array(exp)
    htypes = np.array([types, freq])

    # ilość osób dla poszczególnych kategorii doświadczenia
    perc = np.round(freq / sum(freq) * population).astype(int)

    # cała populacja
    walking = np.repeat(types, perc)

    # ilość zombie
    nselected = np.int(np.round(population * infected))

    # wybór zombie
    zombies = np.random.choice(np.arange(population), nselected, replace=False)
    healthy = np.delete(walking, zombies)
    x = np.repeat(-77, nselected)

    # wstawiamy zombie do populacji
    walking = np.concatenate((healthy, x))

    # procent zombie w populacji
    perc_zombies = np.mean(walking == -77)

    nthings = walking.size
    nthings_vector = np.array([nthings])

    # ilość zombie w czasie
    nzombies = np.array([np.sum(walking == -77)])

    # ilość ludzi dla poszczególnych kategorii
    nhumans = np.array([np.sum(walking != -77)])
    nhumans0 = np.sum(walking == 0)
    nhumans10 = np.sum(walking == 10)
    nhumans20 = np.sum(walking == 20)
    nhumans30 = np.sum(walking == 30)
    nhumans40 = np.sum(walking == 40)
    nhumans50 = np.sum(walking == 50)
    nhumans60 = np.sum(walking == 60)
    nhumans70 = np.sum(walking == 70)
    nhumans80 = np.sum(walking == 80)
    nhumans90 = np.sum(walking == 90)

    logging.debug("Rozkład populacji:  %d %d %d %d %d %d %d %d %d %d" % (nhumans0, nhumans10, nhumans20,
                                                                         nhumans30, nhumans40, nhumans50,
                                                                         nhumans60, nhumans70, nhumans80,
                                                                         nhumans90))

    max_rounds = 100
    i = 0
    # powtarzamy symulacje do max_rounds chyba że wcześniej nie zostanie żaden żywy człowiek albo zombie
    while (nzombies[i] and nhumans[i] and i < max_rounds):
        i += 1

        # tworzymy losowe pary (numery osób)
        encounter = pairup(nthings)

        # pary (doświadczenie, -77 dla zombie)
        types = np.vstack((walking[encounter[:, 0]], walking[encounter[:, 1]])).T

        # rezultaty spotkań
        conflict = np.zeros(types.shape, dtype=int)

        # jeśli mamy parę człowiek - zombie to ją odwracamy
        hvz = (types[:, 1] == -77) & (types[:, 0] >= 0)
        temp = np.copy(types)
        types[hvz, 0] = temp[hvz, 1]
        types[hvz, 1] = temp[hvz, 0]

        # zamieniamy te same pary w tablicy encounter
        temp = np.copy(encounter)
        encounter[hvz, 0] = temp[hvz, 1]
        encounter[hvz, 1] = temp[hvz, 0]

        # znajdujemy pary zombie - człowiek
        zvh = (types[:, 0] == -77) & (types[:, 1] >= 0)

        # prawdopodobieństwo wygranej zombie
        win = (np.random.uniform(size=sum(zvh)) > types[zvh, 1] / 100)

        # zombie wygrywa - kod 4 lub ginie - kod 2
        conflict[zvh, 0] = np.where(win, 4, 2)

        # człowiek wygrywa - kod 6 lub ginie - kod 1
        conflict[zvh, 1] = np.where(win, 1, 6)

        # zombies nie szkodzą sobie - kod 3
        conflict[types[:, 0] == types[:, 1],] = 3

        # ludzie nie szkodzą sobie - kod 3
        conflict[(types[:, 0] >= 0) & (types[:, 1] >= 0),] = 3

        # ludzie, którzy zginęli zamieniają się w zombie - kod 1, albo giną - kod 5
        fallen = np.sum(conflict == 1)
        zombification = (np.random.uniform(size=fallen) < z_ratio)
        conflict[conflict == 1] = np.where(zombification, 1, 5)
        walking[encounter[conflict == 1]] = -77
        logging.debug(
            "Pokonanych ludzi %s, zamiana w zombie %s, umiera %s" % (
                fallen, np.sum(zombification), np.sum(conflict == 5)))
        logging.debug("Gine %s zombie" % np.sum(conflict == 2))

        # ludzie, którzy przeżyli spotkanie z zombie zwiększają doświadczenie 
        if exp_growth:
            winners = encounter[conflict == 6]
            logging.debug("ludzie którzy pokonali zombie %s, doświadczenie %s" % (winners, walking[winners]))
            exp_gain = (np.random.uniform(size=len(winners)) > walking[winners] / 100)
            logging.debug("Ilość osób zwiększających doświadczenie %d" % np.sum(walking[winners[exp_gain]] < 90))
            walking[winners[exp_gain]] = np.minimum(90, walking[winners[exp_gain]] + 10)
            logging.debug("ludzie którzy pokonali zombie %s, doświadczenie %s" % (winners, walking[winners]))

        # pokonane zombie oraz nieżywych i niezamienionych w zombie usuwamy
        walking = np.delete(walking, [encounter[(conflict == 5) | (conflict == 2)]])

        # walking = np.delete(walking, [encounter[conflict == 2]])

        # procent zombie
        perc_zombies = np.append(perc_zombies, np.mean(walking == -77))

        # całkowita populacja (ludzie i zombie)
        nthings = len(walking)
        nthings_vector = np.append(nthings_vector, nthings)

        # ilość zombie
        nzombies = np.append(nzombies, sum(walking == -77))

        # ilość ludzi
        nhumans = np.append(nhumans, sum(walking != -77))

        # ilość ludzi dla poszczególnych wielkości doświadczenia
        nhumans0 = np.append(nhumans0, sum(walking == 0))
        nhumans10 = np.append(nhumans10, sum(walking == 10))
        nhumans20 = np.append(nhumans20, sum(walking == 20))
        nhumans30 = np.append(nhumans30, sum(walking == 30))
        nhumans40 = np.append(nhumans40, sum(walking == 40))
        nhumans50 = np.append(nhumans50, sum(walking == 50))
        nhumans60 = np.append(nhumans60, sum(walking == 60))
        nhumans70 = np.append(nhumans70, sum(walking == 70))
        nhumans80 = np.append(nhumans80, sum(walking == 80))
        nhumans90 = np.append(nhumans90, sum(walking == 90))

        logging.debug("LUDZIE: %s ZOMBIE %s" % (nhumans[-1], nzombies[-1]))
        logging.debug("Rozkład populacji:  %d %d %d %d %d %d %d %d %d %d" % (nhumans0[-1], nhumans10[-1], nhumans20[-1],
                                                                             nhumans30[-1], nhumans40[-1],
                                                                             nhumans50[-1],
                                                                             nhumans60[-1], nhumans70[-1],
                                                                             nhumans80[-1],
                                                                             nhumans90[-1]))

    result = {'humans': nhumans, 'zombies': nzombies, 'humans0': nhumans0, 'humans10': nhumans10, 'humans20': nhumans20,
              'humans30': nhumans30,
              'humans40': nhumans40, 'humans50': nhumans50, 'humans60': nhumans60, 'humans70': nhumans70,
              'humans80': nhumans80, 'humans90': nhumans90}
    return result


def pairup(x, unmatched=True):
    if type(x) == int:
        x = np.arange(x)
    xleng = x.size
    hleng = np.int(np.floor(xleng / 2))
    np.random.shuffle(x)
    if xleng % 2 and unmatched:
        t = x[-1]
        x = np.delete(x, -1)
        x = x.reshape(hleng, 2)
        x = np.vstack((x, (t, t)))
    else:
        x = x.reshape(hleng, 2)
    return x


def statistics(dict):
    df = pd.DataFrame.from_dict(dict)
    humans_p = ['humans' + str(i) for i in range(0, 100, 10)]
    survivors_beg = df.iloc[0][humans_p]
    zombies_beg = df.iloc[0]['zombies']
    survivors = df.iloc[-1][humans_p]
    zombies = df.iloc[-1]['zombies']
    surv_list_beg = [i for i in survivors_beg]
    surv_list = [i for i in survivors]
    logging.info("Początek symulacji:")
    logging.info('Populacja zombi %d' % zombies_beg)
    logging.info('Populacja ludzi %d' % survivors_beg.sum())
    logging.info("Rozkład populacji:  %s" % surv_list_beg)
    logging.info('')
    logging.info("Koniec symulacji")
    logging.info('Populacja zombi %d' % zombies)
    logging.info('Populacja ludzi %d' % survivors.sum())
    logging.info("Rozkład populacji:  %s" % surv_list)


def plot_simulation(dict):
    f = plt.figure()
    af = f.add_subplot(111)
    df = pd.DataFrame.from_dict(dict)
    print(df)
    humans_p = ['humans' + str(i) for i in range(0, 100, 10)]
    af.plot(df['zombies'], label='Z')
    af.plot(df[humans_p])
    for i in range(0, 100, 10):
        af.text(len(df), df.iloc[-1]['humans' + str(i)], i)
    af.text(len(df), df['zombies'][len(df) - 1], "Zombies")
    survivors_beg = df.iloc[0][humans_p]
    surv_list_beg = [i for i in survivors_beg]
    af.set_title("Population: " + str(surv_list_beg))
    plt.show(block=False)


def gaussian(x, x0, sigma):
    return np.exp(-np.power((x - x0) / sigma, 2.) / 2.)


logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

# test dla rozkładów normalnych o różnej wartości średniej (populacja symetryczna to średnia 45)
for skew in [10, 20, 30, 40, 50, 60]:
    exp = [gaussian(x, skew, 20) * 100 for x in range(0, 100, 10)]
    logging.debug(exp)
    d = sim(population=100000, infected=0.02, exp=exp, exp_growth=True)
    statistics(d)
    plot_simulation(d)
plt.show()
