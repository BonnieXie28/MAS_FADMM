project_root = fileparts(fileparts(mfilename('fullpath')));
Data = xlsread(fullfile(project_root,'visual_gamma.xlsx'),'m4_f4');

Data1 = Data(:,1:4);
Data2 = Data(:,5:8);

m=4; row = 2; col = 2;

x_2 = linspace(0,1,3);
x_3 = linspace(0,1,4);
x_4 = linspace(0,1,5);

data1 = Data1(1:3,:); data11 = Data2(1:3,:);
data2 = Data1(4:7,:); data22 = Data2(4:7,:);
data3 = Data1(8:12,:);data33 = Data2(8:12,:);

u_data1 = cumsum(data1); u_data11 = cumsum(data11);
u_data2 = cumsum(data2); u_data22 = cumsum(data22);
u_data3 = cumsum(data3); u_data33 = cumsum(data33);

max_u_1 = max(u_data1);min_u_1 = min(u_data1);
max_u_2 = max(u_data2);min_u_2 = min(u_data2);
max_u_3 = max(u_data3);min_u_3 = min(u_data3);

u_scale_1 = (u_data1 - min_u_1)/sum(max_u_1-min_u_1);
u_scale_2 = (u_data2 - min_u_2)/sum(max_u_2-min_u_2);
u_scale_3 = (u_data3 - min_u_3)/sum(max_u_3-min_u_3);

j=1;
subplot(row,col,j);
y_1 = u_scale_1(:,j);
y_2 = u_scale_2(:,j);
y_3 = u_scale_3(:,j);
y_4 = u_data11(:,j);
y_5 = u_data22(:,j);
y_6 = u_data33(:,j);

A=plot(x_2, y_1, 'o-r');
hold on;
B=plot(x_3, y_2, 'o-g');
hold on;
C=plot(x_4, y_3, 'o-b');
hold on;
D=plot(x_2, y_4, '*--r');
hold on;
E=plot(x_3, y_5, '*--g');
hold on;
F=plot(x_4, y_6, '*--b');
title('(a)','FontSize',10)
xlabel('x_1','FontSize',10)
ylabel('u_1','FontSize',10)
lgd = legend([A,B,C,D,E,F],{'FADMM \gamma = 2',' UTADIS \gamma = 2', 'FADMM \gamma = 3', 'UTADIS \gamma = 3', 'FADMM \gamma = 4', ' UTADIS \gamma = 4'},'orientation','horizontal','Location','southoutside');
set(lgd,'Units', 'normalized','position',[0.35 0 0.35 0.05],'fontsize',10);


j=2;
subplot(row,col,j);
y_1 = u_scale_1(:,j);
y_2 = u_scale_2(:,j);
y_3 = u_scale_3(:,j);
y_4 = u_data11(:,j);
y_5 = u_data22(:,j);
y_6 = u_data33(:,j);

A=plot(x_2, y_1, 'o-r');
hold on;
B=plot(x_3, y_2, 'o-g');
hold on;
C=plot(x_4, y_3, 'o-b');
hold on;
D=plot(x_2, y_4, '*--r');
hold on;
E=plot(x_3, y_5, '*--g');
hold on;
F=plot(x_4, y_6, '*--b');
title('(b)','FontSize',10)
xlabel('x_2','FontSize',10)
ylabel('u_2','FontSize',10)

lgd = legend([A,B,C,D,E,F],{'FADMM \gamma = 2',' UTADIS \gamma = 2', 'FADMM \gamma = 3', 'UTADIS \gamma = 3', 'FADMM \gamma = 4', ' UTADIS \gamma = 4'},'orientation','horizontal','Location','southoutside');
set(lgd,'Units', 'normalized','position',[0.35 0 0.35 0.05],'fontsize',10);

j=3;
subplot(row,col,j);
y_1 = u_scale_1(:,j);
y_2 = u_scale_2(:,j);
y_3 = u_scale_3(:,j);
y_4 = u_data11(:,j);
y_5 = u_data22(:,j);
y_6 = u_data33(:,j);

A=plot(x_2, y_1, 'o-r');
hold on;
B=plot(x_3, y_2, 'o-g');
hold on;
C=plot(x_4, y_3, 'o-b');
hold on;
D=plot(x_2, y_4, '*--r');
hold on;
E=plot(x_3, y_5, '*--g');
hold on;
F=plot(x_4, y_6, '*--b');
title('(c)','FontSize',10)
xlabel('x_3','FontSize',10)
ylabel('u_3','FontSize',10)

lgd = legend([A,B,C,D,E,F],{'FADMM \gamma = 2',' UTADIS \gamma = 2', 'FADMM \gamma = 3', 'UTADIS \gamma = 3', 'FADMM \gamma = 4', ' UTADIS \gamma = 4'},'orientation','horizontal','Location','southoutside');
set(lgd,'Units', 'normalized','position',[0.35 0 0.35 0.05],'fontsize',10);

j=4;
subplot(row,col,j);
y_1 = u_scale_1(:,j);
y_2 = u_scale_2(:,j);
y_3 = u_scale_3(:,j);
y_4 = u_data11(:,j);
y_5 = u_data22(:,j);
y_6 = u_data33(:,j);

A=plot(x_2, y_1, 'o-r');
hold on;
B=plot(x_3, y_2, 'o-g');
hold on;
C=plot(x_4, y_3, 'o-b');
hold on;
D=plot(x_2, y_4, '*--r');
hold on;
E=plot(x_3, y_5, '*--g');
hold on;
F=plot(x_4, y_6, '*--b');
title('(d)','FontSize',10)
xlabel('x_4','FontSize',10)
ylabel('u_4','FontSize',10)

lgd = legend([A,B,C,D,E,F],{'FADMM \gamma = 2',' UTADIS \gamma = 2', 'FADMM \gamma = 3', 'UTADIS \gamma = 3', 'FADMM \gamma = 4', ' UTADIS \gamma = 4'},'orientation','horizontal','Location','southoutside');
set(lgd,'Units', 'normalized','position',[0.35 0 0.35 0.05],'fontsize',10);
