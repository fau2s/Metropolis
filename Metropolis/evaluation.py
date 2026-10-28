def evaluation(self, V, Pperf, Tperf, x, portfheld, C, C0, numtrade):
    Cfinal = 0
    V[Tperf] = 0
    if portfheld:
        # print('CASH IN at the   .')
        Cfinal = C[Tperf - 1] + x[Tperf, :].dot(Pperf[:, Tperf - 1])
    else:
        Cfinal = C[Tperf]

    Return = 100 * (Cfinal - C0) / C0
    # print(['Final cash is ', str(Cfinal)])
    # print(['A return of ', str(Return), ' #'])
    # print(['We made ', str(numtrade), ' trades.'])
    profit = Cfinal - C0
    numTrade = numtrade
    # pyplot.plot(range(Tperf+1), C + V, range(Tperf+1), C)
    # pyplot.gcf().clear()
    # pyplot.hist(Return,30)
    # figure
    # Plot orig portfolio value and mu estimate
    # plot (10:Tid+Tperf,xoptbuy0'*P_all(:,10:Tid+Tperf),Tid+1:Tid+Tperf,mu)
    # Plot buy and sell pointspyplot.plot(range(Tperf), C + V, range(Tperf+1), C)
    # hold on
    # plot (buyt,buyp,'rs',sellt,sellp,'rh')
    # hold off
    # leg  ('Portfolio Value','Mean Estimate','Buy Action',...
    #     'Sell Action', 'Location', 'SouthWe
    #

    return Return, profit, numTrade