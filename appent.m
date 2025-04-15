function [app_totent,app_ent,app_prob] = appent(data)
app_prob=data./sum(data,"omitnan");
app_ent=-(app_prob.*log2(app_prob));
app_totent=sum(app_ent,"omitnan");
end