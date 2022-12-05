import com.comsol.model.*
import com.comsol.model.util.*

model = mphopen('Optimization_of_tube_heattransfer.mph');

velocity_range = 10: 10: 60;  % define the range of the initial velocity

figure(1);

for i = 1: 1: length(velocity_range)
    disp(velocity_range(i))
    model.component('comp1').variable('var1').set('V_ini', [num2str(velocity_range(i)/1000),'[m/s]']);
    % model.param.set('V_ini', join(num2str(velocity_range),'[mm/s]') );
    model.study('std1').run;
    subplot(3,4,2*i-1);
    mphplot(model,"pg1");
    subplot(3,4,2*i);
    mphplot(model,"pg2");
end
