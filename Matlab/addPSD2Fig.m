function PSD_visits=addPSD2Fig(params,PSD_visits,psd_fig,c)

PSD_visitFlag=1;

if PSD_visitFlag
    %plot OFF
    figure(PSD_visits) ;set(PSD_visits, 'Visible', 'off');
    hold on
       
        % change PSD color
        axesObjs = get(psd_fig, 'Children');  %axes handles
        dataObjs = get(axesObjs, 'Children'); %handles to low-level graphics objects in axes
        set(dataObjs,'Color',c)

        L = findobj(psd_fig(1),'type','line');
        copyobj(L,findobj(PSD_visits,'type','axes'));
        
        
        %Detiails
        title('PSD (Blue-Off, Green-On)')
        if params.zScoreFlag==1
            ylabel('z^2/hz')
        end
         if params.zScoreFlag==0
            ylabel('v^2/hz')
         end
        
         title ('PSD per visit')
%          xlim([0 140]);
        
        
    end
end
