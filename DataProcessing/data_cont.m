MAXlabelnumber=10;
xlist=[1,4,7,10,13,16,19,22,25,28,31,34,37,40,43,46,49,52,55,58,61,64];
exp=3;
foldername='cont_data_zeronormalize_ex/';
imageske=cell(56*20,1);%Original
worldske=cell(56*20,1);%
labelall=cell(56*20,1);
labelnumall=cell(56*20,1);
lengthall=cell(56*20,1);
section=cell(56*20,1);
imageske_test=cell(14*20,1);%5 th essai as testing
worldske_test=cell(14*20,1);%
labelall_test=cell(14*20,1);
labelnumall_test=cell(14*20,1);
lengthall_test=cell(14*20,1);
section_test=cell(14*20,1);
A=[1 2 3 4]';
A=repmat(A,14,1);
B= linspace(1,14,14);
B= repmat(B,4,1);
B=reshape(B,[4*14,1]);
index=[B,A];
%index=[index;index];
%finger=[ones(56,1);ones(56,1)*2];
%index=[index,finger];
index=repmat(index,MAXlabelnumber,1);% each data most take part in for MAXlabelnumber/2 times

info=load('informations_troncage_sequences.txt');
indexmother=1;

for subject=1:20
    trainind=randperm(56*MAXlabelnumber);
        indexnow=1;
         for seq=1:56
                labelnumber=randi([3,MAXlabelnumber]);
                allske=[];
                allworld=[];
                label=[];
                lengths=0;
                last=[];
                lastw=[];
                sec=[];
                    for i =1:labelnumber
                    gesture=index(trainind(indexnow),1);
                    essai=index(trainind(indexnow),2);
                   % fingerstyle=index(trainind(indexnow),3);
                    target=[gesture 2 subject essai];% gesture  finger subject essai 14 2 20 5          
                    path=['gesture_', num2str(target(1)) ,'\finger_',num2str(target(2)),'\subject_',num2str(target(3)),'\essai_',num2str(target(4)),'\'];
                    general=load([path ,'general_information.txt' ]);
                    nowl=length(general);
                    
                    skeleton=load([path,'skeleton_image.txt']);
                    world=load([path,'skeleton_world.txt']);

                    ind=find(info(:,1)==target(1) & info(:,2)==target(2) & info(:,3)==target(3) & info(:,4)==target(4));
                    
                     final=info(ind,6)+min(5,nowl-info(ind,6));                 
                    start=3; 
                    if gesture==3
                       nowlb=nowl;
                       before=world(1:info(ind,5),:);
                       after=world(info(ind,5):end,:);
                       newbefore=downsample(before,2);
                       nowl=length(newbefore(:,1))+length(after(:,1));
                       world=[newbefore;after];
                       newend=info(ind,6)-(nowlb-nowl);
                       final=newend+min(5,nowl-newend);   
                       if nowl-newend<0
                          wrong 
                       end
                    end
                    
                   
                    world(:,xlist)=world(:,xlist)-repmat(world(1,1),[nowl,22]);
                    world(:,xlist+1)=world(:,xlist+1)-repmat(world(1,2),[nowl,22]);
                    world(:,xlist+2)=world(:,xlist+2)-repmat(world(1,3),[nowl,22]);
                    
                       
                    
                   
                   
                    
                  
                     frame=final-start+1;
                    lengths=lengths+frame;
                    
                                   %% Conntection
                    if i ~=1
                         insert=randi([10,15]);
                       lengths=lengths+insert;
                       dif=(skeleton(start,:)-last)/insert;
                       difw=(world(start,:)-lastw)/insert;
                       for k=1:insert
                          last=last+dif;
                          lastw=lastw+difw;
                          allske=[allske;last];
                          allworld=[allworld;lastw];
                       end
                    end
                                   
                                   
                                   %%
                    last=skeleton(final,:);
                    lastw = world(final,:);
                    allske=[allske;skeleton(start:final,:)];
                    allworld=[allworld;world(start:final,:)];
                    label=[label,gesture-1];
                    sec=[sec;[lengths-frame,lengths-1]];%zero start coding
                    indexnow=indexnow+1;
                    end        
                imageske{indexmother,1}=allske;
                worldske{indexmother,1}=allworld;
                labelall{indexmother,1}=label;
                lengthall{indexmother,1}=lengths;
                labelnumall{indexmother,1}=labelnumber;
                section{indexmother,1}=sec;
                indexmother
                indexmother=indexmother+1;
         end
              
end
%}
A=[5];
A=repmat(A,14,1);
B= linspace(1,14,14)';
%B= repmat(B,4,1);
%B=reshape(B,[4*14,1]);
index=[B,A];
%index=[index;index];
%finger=[ones(14,1);ones(14,1)*2];
%index=[index,finger];
index=repmat(index,MAXlabelnumber,1);
indexmother=1;
for subject=1:20
    trainind=randperm(14*MAXlabelnumber);
        indexnow=1;
         for seq=1:14
              labelnumber=randi([3,MAXlabelnumber]);
               sec=[];
                allske=[];
                allworld=[];
                label=[];
                lengths=0;
                last=[];
                lastw=[];
                 for i =1:labelnumber
                    gesture=index(trainind(indexnow),1);
                    essai=index(trainind(indexnow),2);
                   % fingerstyle=index(trainind(indexnow),3);
                    target=[gesture 2 subject essai];% gesture  finger subject essai 14 2 20 5          
                    path=['gesture_', num2str(target(1)) ,'\finger_',num2str(target(2)),'\subject_',num2str(target(3)),'\essai_',num2str(target(4)),'\'];
                    general=load([path ,'general_information.txt' ]);
                    nowl=length(general);
                    skeleton=load([path,'skeleton_image.txt']);
                    world=load([path,'skeleton_world.txt']);

                    ind=find(info(:,1)==target(1) & info(:,2)==target(2) & info(:,3)==target(3) & info(:,4)==target(4));
                    start=info(ind,5)-5;
                    if start <=0
                       start=1; 
                    end
                  
                        
                    start=3; 
                   
                    world(:,xlist)=world(:,xlist)-repmat(world(1,1),[nowl,22]);
                    world(:,xlist+1)=world(:,xlist+1)-repmat(world(1,2),[nowl,22]);
                    world(:,xlist+2)=world(:,xlist+2)-repmat(world(1,3),[nowl,22]);
                    
                 
                    
                    final=info(ind,6);
                   
                        final=nowl;
                   
                    frame=final-start+1;
                    lengths=lengths+frame;
                    
                                   %% Conntection
                    if i ~=1
                         insert=randi([10,15]);
                       lengths=lengths+insert;
                       dif=(skeleton(start,:)-last)/insert;
                       difw=(world(start,:)-lastw)/insert;
                       for k=1:insert
                          last=last+dif;
                          lastw=lastw+difw;
                          allske=[allske;last];
                          allworld=[allworld;lastw];
                       end
                    end
                                   
                                   
                                   %%
                    last=skeleton(final,:);
                    lastw = world(final,:);
                    allske=[allske;skeleton(start:final,:)];
                    allworld=[allworld;world(start:final,:)];
                    label=[label,gesture-1];
                    sec=[sec;[lengths-frame,lengths-1]];%zero start coding
                    indexnow=indexnow+1;
                    end        
                imageske_test{indexmother,1}=allske;
                worldske_test{indexmother,1}=allworld;
                labelall_test{indexmother,1}=label;
                lengthall_test{indexmother,1}=lengths;
                labelnumall_test{indexmother,1}=labelnumber;
                 section_test{indexmother,1}=sec;
                indexmother
                indexmother=indexmother+1;
         end
              
end


imageske_exp=cell(56*20*exp,1);%Original
worldske_exp=cell(56*20*exp,1);%
labelall_exp=cell(56*20*exp,1);
lengthall_exp=cell(56*20*exp,1);
labelnumall_exp=cell(56*20*exp,1);
section_exp=cell(56*20*exp,1);
indexexp=1;
for i=1:56*20
                frame=lengthall{i,1}(1,1);
                secnow=section{i,1};
                oris=worldske{i,1};
                oris=reshape(oris,[frame,3,22]);
                labelori=labelall{i,1};
                labelnum=labelnumall{i,1};
                for expand=1:exp
                
                lengthall_exp{indexexp,1}=frame;
                labelnumall_exp{indexexp,1}=labelnum;
                temp=oris;
                if expand==1%original
                    thetax=0;
                    thetay=0;
                    thetaz=0;
                    
                else% random rotate -20~20 degree by x, y ,z axises 
                    thetax=rand(1)*20-10;
                    thetay=rand(1)*20-10;
                    thetaz=rand(1)*20-10;
                end
                
                rotation3dx=[1 0 0;0 cosd(thetax) -sind(thetax); 0 sind(thetax) cosd(thetax)];
                rotation3dy=[cosd(thetay) 0 sind(thetay);0 1 0;-sind(thetay) 0 cosd(thetay)];
                rotation3dz=[cosd(thetaz) -sind(thetaz) 0;sind(thetaz) cosd(thetaz) 0;0 0 1];
                for L=1: frame%need to rotate frame by frame
                    tempL=reshape(temp(L,:,:),[3,22]);
                    
                    tempL=rotation3dz*rotation3dy*rotation3dx*tempL;
                    temp(L,:,:)=reshape(tempL,[1,3,22]);
                end
                temp=reshape(temp,[frame,66]);
                worldske_exp{indexexp,1}=temp;
                labelall_exp{indexexp,1}=labelori;
                section_exp{indexexp,1}=secnow;
                indexexp=indexexp+1;
                indexexp
               end
             
    
end

save([foldername 'imageske.mat'],'imageske_exp');
save([foldername 'labelall.mat'],'labelall_exp');
save([foldername 'labelnumall.mat'],'labelnumall_exp');
save([foldername 'lengthall.mat'],'lengthall_exp');
save([foldername 'section.mat'],'section_exp');
save([foldername 'worldske.mat'],'worldske_exp');

save([foldername 'test/imageske.mat'],'imageske_test');
save([foldername 'test/labelall.mat'],'labelall_test');
save([foldername 'test/labelnumall.mat'],'labelnumall_test');
save([foldername 'test/lengthall.mat'],'lengthall_test');
save([foldername 'test/section.mat'],'section_test');
save([foldername 'test/worldske.mat'],'worldske_test');