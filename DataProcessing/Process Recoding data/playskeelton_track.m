function no = playskeelton( world2,start,xmat,ymat,zmat,cc)
  temp=world2;
  temp([1,4,7,10,13,16,19,22,25,28,31,34,37,40,43,46,49,52,55,58,61,64])=temp([1,4,7,10,13,16,19,22,25,28,31,34,37,40,43,46,49,52,55,58,61,64])-repmat(start(1),[1,22]);
  temp([1,4,7,10,13,16,19,22,25,28,31,34,37,40,43,46,49,52,55,58,61,64]+1)=temp([1,4,7,10,13,16,19,22,25,28,31,34,37,40,43,46,49,52,55,58,61,64]+1)-repmat(start(2),[1,22]);
  temp([1,4,7,10,13,16,19,22,25,28,31,34,37,40,43,46,49,52,55,58,61,64]+2)=temp([1,4,7,10,13,16,19,22,25,28,31,34,37,40,43,46,49,52,55,58,61,64]+2)-repmat(start(3),[1,22]);
  world2=temp;
   
   a=world2([1 4]);
   b=world2([2 5]);
   c=world2([3,6]);
  % subplot(1,2,2)
  %temp=rotation3dz*rotation3dy*rotation3dx*reshape(temp,[3,22]);
  plot3(c,a,b,'LineWidth',5)
   xlabel('depth');
 ylabel('x');
 zlabel('y');
   view([-90 -0]);
     axis([-0.2 0.2 -0.23 0.23  -0.2  0.2])
     set(gca,'Ydir','reverse')

   hold on
  % x=skeleton(i,[1 3 5 7 9 11 13 15 17 19 21 23 25 27 29 31 33 35 37 39 41 43]);
  % y=skeleton(i,[2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44]);
% scatter3(zmat,xmat,ymat,30,cc,'filled')
%   plot3(zmat,xmat,ymat,'red')
  
   x=world2([4 7 10 13 16 ]);
   y=world2([4 7 10 13 16 ]+1);
   z=world2([4 7 10 13 16 ]+2);
   %scatter(x,y,30,'r','filled');
  
   plot3(z,x,y,'LineWidth',5);
   x=world2([4 19 22 25 28]);
   y=world2([4 19 22 25 28]+1);
   z=world2([4 19 22 25 28]+2);
 
   plot3(z,x,y,'LineWidth',5);
 
   x=world2([4 31 34 37 40]);
   y=world2([4 31 34 37 40]+1);
   z=world2([4 31 34 37 40]+2);
  
   plot3(z,x,y,'LineWidth',5);
    
   x=world2([4 43 46 49 52]);
   y=world2([4 43 46 49 52]+1);
   z=world2([4 43 46 49 52]+2);
  
   plot3(z,x,y,'LineWidth',5);
  
   x=world2([4 55 58 61 64]);
   y=world2([4 55 58 61 64]+1);
    z=world2([4 55 58 61 64]+2);
  
   plot3(z,x,y,'LineWidth',5);
  
 % pbaspect([1 1 0.001])
   hold off 
end

