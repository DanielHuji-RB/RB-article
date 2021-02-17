%Daniel Sand
function  subT= subTable(T,colReference,objectToRowCompare,col)

if (nargin<4)
    col=':';
end


if isa(colReference,'double') & isa(objectToRowCompare,'double') %double case
    rows = num2str(colReference)==num2str(objectToRowCompare)
else
    rows =strcmp(colReference,objectToRowCompare); %char case
end
subT=T(rows,col); %T_Off(:,scoreReference)


end