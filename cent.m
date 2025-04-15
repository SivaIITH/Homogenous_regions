function [tot_ent,hiscon,width,prob,h1,h2h3]=cent(data)
edges=[0 10 35.5 64.4 124.4 244.4 inf]; % The bin widths are considered based on Rajeevan et al.,2020
[hiscon,width]=histcounts(data,edges);
hiscon=hiscon+0.01;   % Adding 0.01 to avaoid the nan values in the entropy.
% Adding 0.01 dont have any significant impact on entropy values.
prob=hiscon/sum(hiscon);
ent=-(prob.*log2(prob));
h1=ent(1,1);
h2h3=sum(ent(1,2:end));
tot_ent=h1+h2h3;
end