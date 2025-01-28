%%
cx=readmatrix('dt_tals_cl.csv','Range','A2:AQ5708');
y0=48*ones(size(cx,1),1);
ncl=4;
if (ncl==2)
    Colcl=4;
end
if (ncl==3)
    Colcl=5;
end
if (ncl==4)
    Colcl=6;
end
if (ncl==5)
    Colcl=7;
end
it30=8;
% sztp(cx(:,it30:end),'sztals.csv');
figure()
subplot(121)
hold on
plot_traj([y0 cx(:,it30:end)],cx(:,Colcl),1,'Total ALSFRS',0.5,0.5,1,0.3,'traj1TALS.csv');
hold on
plot_traj([y0 cx(:,it30:end)],cx(:,Colcl),2,'Total ALSFRS',0,0,1,0.7,'traj2TALS.csv');
hold on
if (ncl>=3)
plot_traj([y0 cx(:,it30:end)],cx(:,Colcl),3,'Total ALSFRS',0,1,1,0.3,'traj3TALS.csv');
end
if (ncl>=4)
plot_traj([y0 cx(:,it30:end)],cx(:,Colcl),4,'Total ALSFRS',1,1,0,0.3,'traj4TALS.csv');
end
if (ncl>=5)
plot_traj([y0 cx(:,it30:end)],cx(:,Colcl),5,'Total ALSFRS',1,0,1,0.3,'traj5TALS.csv');
end
subplot(122)
hold on
plotsurvcurve(cx(:,Colcl),1,cx(:,2),cx(:,3),0.5,0.5,1,0.3,'surv1TALS.csv');
plotsurvcurve(cx(:,Colcl),2,cx(:,2),cx(:,3),0,0,1,0.7,'surv2TALS.csv');
if(ncl>=3)
plotsurvcurve(cx(:,Colcl),3,cx(:,2),cx(:,3),0,1,1,0.3,'surv3TALS.csv');
end
if(ncl>=4)
plotsurvcurve(cx(:,Colcl),4,cx(:,2),cx(:,3),1,1,0,0.3,'surv4TALS.csv');
end
if(ncl>=5)
plotsurvcurve(cx(:,Colcl),5,cx(:,2),cx(:,3),1,0,1,0.3,'surv5TALS.csv');
end


%%
cx=readmatrix('dt_bulb_cl.csv','Range','A2:AQ5456');
y0=12*ones(size(cx,1),1);
ncl=4;
if (ncl==2)
    Colcl=4;
end
if (ncl==3)
    Colcl=5;
end
if (ncl==4)
    Colcl=6;
end
if (ncl==5)
    Colcl=7;
end
it30=8;
 %sztp(cx(:,it30:end),'tmp.csv');
figure()
subplot(121)
hold on
plot_traj([y0 cx(:,it30:end)],cx(:,Colcl),1,'Bulbar score',0.5,0.5,1,0.3,'traj1Bulb.csv');
hold on
plot_traj([y0 cx(:,it30:end)],cx(:,Colcl),2,'Bulbar score',0,0,1,0.7,'traj2Bulb.csv');
hold on
if (ncl>=3)
plot_traj([y0 cx(:,it30:end)],cx(:,Colcl),3,'Bulbar score',0,1,1,0.7,'traj3Bulb.csv');
hold on
end
if (ncl>=4)
plot_traj([y0 cx(:,it30:end)],cx(:,Colcl),4,'Bulbar score',0,1,0,0.8,'traj4Bulb.csv');
hold on
end
if (ncl>=5)
plot_traj([y0 cx(:,it30:end)],cx(:,Colcl),5,'Bulbar score',1,1,0,0.8,'traj5Bulb.csv');
hold on
end
ylim([0 12]);
subplot(122)
hold on
plotsurvcurve(cx(:,Colcl),1,cx(:,2),cx(:,3),0.5,0.5,1,0.3,'surv1Bulb.csv');
plotsurvcurve(cx(:,Colcl),2,cx(:,2),cx(:,3),0,0,1,0.7,'surv2Bulb.csv');
if (ncl>=3)
    plotsurvcurve(cx(:,Colcl),3,cx(:,2),cx(:,3),0,1,1,0.7,'surv3Bulb.csv');
end
if (ncl>=4)
    plotsurvcurve(cx(:,Colcl),4,cx(:,2),cx(:,3),0,1,0,0.7,'surv4Bulb.csv');
end
if (ncl>=5)
    plotsurvcurve(cx(:,Colcl),5,cx(:,2),cx(:,3),1,1,0,0.7,'surv5Bulb.csv');
end

%%
cx=readmatrix('dt_q3_cl.csv','Range','A2:AP5456');
y0=4*ones(size(cx,1),1);
ncl=3;
if (ncl==2)
    Colcl=4;
end
if (ncl==3)
    Colcl=5;
end
if (ncl==4)
    Colcl=6;
end
it30=7;
%sztp(cx(:,it30:end),'tmp.csv');
figure()
subplot(121)
hold on
plot_traj([y0 cx(:,it30:end)],cx(:,Colcl),1,'q3 score',0.5,0.5,1,0.3,'traj1q3.csv');
hold on
plot_traj([y0 cx(:,it30:end)],cx(:,Colcl),2,'q3 score',0,0,1,0.7,'traj2q3.csv');
hold on
if (ncl>=3)
plot_traj([y0 cx(:,it30:end)],cx(:,Colcl),3,'q3 score',0,1,1,0.7,'traj3q3.csv');
hold on
end
if (ncl>=4)
plot_traj([y0 cx(:,it30:end)],cx(:,Colcl),4,'q3 score',0,1,0,0.8,'traj4q3.csv');
hold on
end
if (ncl>=5)
plot_traj([y0 cx(:,it30:end)],cx(:,Colcl),5,'q3 score',1,1,0,0.8,'traj5q3.csv');
hold on
end
subplot(122)
hold on
plotsurvcurve(cx(:,Colcl),1,cx(:,2),cx(:,3),0.5,0.5,1,0.3,'surv1q3.csv');
plotsurvcurve(cx(:,Colcl),2,cx(:,2),cx(:,3),0,0,1,0.7,'surv2q3.csv');
if (ncl>=3)
    plotsurvcurve(cx(:,Colcl),3,cx(:,2),cx(:,3),0,1,1,0.7,'surv3q3.csv');
end
if (ncl>=4)
    plotsurvcurve(cx(:,Colcl),4,cx(:,2),cx(:,3),0,1,0,0.7,'surv4q3.csv');
end
if (ncl>=5)
    plotsurvcurve(cx(:,Colcl),5,cx(:,2),cx(:,3),1,1,0,0.7,'surv5q3.csv');
end
 
 
%%
cx=readmatrix('dt_Resp_cl.csv','Range','A2:AP2953');
ncl=3;
if (ncl==2)
    Colcl=4;
end
if (ncl==3)
    Colcl=5;
end
if (ncl==4)
    Colcl=6;
end
it30=7;
sztp(cx(:,it30:end),'tmp.csv');

y0=100*ones(size(cx,1),1);
figure()
subplot(121)
hold on
plot_traj([y0 cx(:,it30:end)],cx(:,Colcl),1,'Respration Marker',0.5,0.5,1,0.3,'traj1Resp.csv');
hold on
plot_traj([y0 cx(:,it30:end)],cx(:,Colcl),2,'Respiration Marker',0,0,1,0.7,'traj2Resp.csv');
hold on
if (ncl>=3)
plot_traj([y0 cx(:,it30:end)],cx(:,Colcl),3,'Respiration Marker',0,1,1,0.3,'traj3Resp.csv');
end
hold on
if (ncl==4)
plot_traj([y0 cx(:,it30:end)],cx(:,Colcl),4,'Respiration Marker',1,0,1,0.3,'traj4Resp.csv');
end
ylim([0 150])
subplot(122)
hold on
plotsurvcurve(cx(:,Colcl),1,cx(:,2),cx(:,3),0.5,0.5,1,0.3,'surv1Resp.csv');
hold on
plotsurvcurve(cx(:,Colcl),2,cx(:,2),cx(:,3),0,0,1,0.7,'surv2Resp.csv');
if (ncl>=3)
plotsurvcurve(cx(:,Colcl),3,cx(:,2),cx(:,3),0,1,1,0.3,'surv3Resp.csv');
end
hold on
if (ncl>=4)
plotsurvcurve(cx(:,Colcl),4,cx(:,2),cx(:,3),1,0,1,0.3,'surv4Resp.csv');
end
%survcurve_export95confInt(cx(:,Colcl),1,cx(:,2),cx(:,3),0.5,0.5,1,0.3,'RespSurvCl1.xlsx')
%survcurve_export95confInt(cx(:,Colcl),2,cx(:,2),cx(:,3),0.5,0.5,1,0.3,'RespSurvCl2.xlsx')
%survcurve_export95confInt(cx(:,Colcl),2,cx(:,2),cx(:,3),0.5,0.5,1,0.3,'RespSurvCl3.xlsx')



%% 
% wireplots
figure
nt=7,%%%%colun from which traj info starts
nclasses=3;%%%%number of classes
cl1=5;%%%%colmn of classes
id1=find(cx(:,cl1)==1);
plot([30:30:1080],[cx(id1,nt:end)],'color',[0 0 1 0.5])
mn1=-9999*ones(36,1);
for i=1:36
idmn1=find(cx(:,cl1)==1 & isnan(cx(:,i+nt-1))==0);
mn1(i)=mean(cx(idmn1,i+nt-1));
end
hold on
plot([30:30:1080],mn1,'color',[0 0 1],'LineWidth',2)
hold on
id2=find(cx(:,cl1)==2);
plot([30:30:1080],[cx(id2,nt:end)],'color',[0.93 0.69 .13 0.5])
mn2=-9999*ones(36,1);
for i=1:36
idmn2=find(cx(:,cl1)==2 & isnan(cx(:,i+nt-1))==0);
mn2(i)=mean(cx(idmn2,i+nt-1));
end
plot([30:30:1080],mn2,'color',[0.93 0.69 0.13],'LineWidth',2)

if (nclasses==3)
    hold on
   
id3=find(cx(:,cl1)==3);
plot([30:30:1080],[cx(id3,nt:end)],'color',[1 0 .0])
mn3=-9999*ones(36,1);
for i=1:36
idmn3=find(cx(:,cl1)==3 & isnan(cx(:,i+nt-1))==0);
mn3(i)=mean(cx(idmn3,i+nt-1));
end
plot([30:30:1080],mn3,'color',[1 0 0],'LineWidth',2)
end

