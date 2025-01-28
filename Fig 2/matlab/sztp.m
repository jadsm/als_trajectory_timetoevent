function sztp(cxn,fname)
sz=[];
for i=1:36
sz=[sz;size(find(isnan(cxn(:,i))==0),1)];
end
writematrix(sz,fname)