file=3;
frame=csvread(['file' num2str(file) '\spec.csv']);
skeleton=csvread(['file' num2str(file) '\joint.csv']);
start=skeleton(1,1:3);
color=linspace(1,0,frame(2));
xmat=[];
ymat=[];
zmat=[];
cc=[];

xmat2=[];
ymat2=[];
zmat2=[];
cc2=[];
fig=figure('Name','gesture','Position',[200,200,500,500]);
%set(fig, 'visible', 'off');
image=cell(frame(2),1);
for i=frame(1):frame(1)+frame(2)-1
    relative_hand_frame=i-frame(1);
  
   %subplot(1,2,1)
   % pic=imread(['file' num2str(file) '\depth\hi' num2str(i+1) 'rgb.png']);
   % nowf=im2double(pic);
    %image{relative_hand_frame+1,1}=nowf;
   % imshow(nowf);
    
   xmat=[xmat;skeleton(relative_hand_frame+1,4)-start(1)];
   ymat=[ymat;skeleton(relative_hand_frame+1,5)-start(2)];
   zmat=[zmat;skeleton(relative_hand_frame+1,6)-start(3)];
   cc=[cc;[1,color(relative_hand_frame+1),color(relative_hand_frame+1)]];
   playskeelton_track(skeleton(relative_hand_frame+1,:),start,xmat,ymat,zmat,cc);
 %{   
     subplot(1,2,2)
    xmat2=[xmat2;skeleton(relative_hand_frame+1,1)-start(1)];
   ymat2=[ymat2;skeleton(relative_hand_frame+1,2)-start(2)];
   zmat2=[zmat2;skeleton(relative_hand_frame+1,3)-start(3)];
   cc2=[cc2;[1,color(relative_hand_frame+1),color(relative_hand_frame+1)]];
    playskeelton_track(skeleton(relative_hand_frame+1,:),start,xmat2,ymat2,zmat2,cc2);
%}    
   pause(1/30)
  %saveas(fig,['file999998/ske' num2str(relative_hand_frame)  '.png']);

end