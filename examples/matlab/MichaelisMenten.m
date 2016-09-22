function v = MichaelisMenten(parameters, independent)

    Vmax = parameters(1);
    Km = parameters(2);
    
    S = independent(:,1);
    
    v = Vmax.*S./(Km + S);