function [cm] = centri(data1,ann_rain_data)
for j = 1:size(data1,2)
    for k = 1:size(data1,1)
        pro(k,j)=k.*data1(k,j);
    end
end
cs=sum(pro);
for i = 1:length(cs)
cm(1,i)=floor(cs(1,i)./ann_rain_data(1,i));
end
end