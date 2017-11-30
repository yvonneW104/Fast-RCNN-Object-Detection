function chosen_detection = nms (bbox_prob, nms_thred)
    x1 = bbox_prob(:,1);
    y1 = bbox_prob(:,2);
    x2 = bbox_prob(:,3);
    y2 = bbox_prob(:,4);
    prob = bbox_prob(:,end);

    area = (x2-x1+1) .* (y2-y1+1);
    
    [~, index] = sort(prob);

    chosen = prob*0;
    count = 1;
    while ~isempty(index)
        last = length(index);
        i = index(last);
        chosen(count) = i;
        count = count + 1;

        xx1 = max(x1(i), x1(index(1:last-1)));
        yy1 = max(y1(i), y1(index(1:last-1)));
        xx2 = min(x2(i), x2(index(1:last-1)));
        yy2 = min(y2(i), y2(index(1:last-1)));

        w = max(0.0, xx2-xx1+1);
        h = max(0.0, yy2-yy1+1);

        inter = w.*h;
        o = inter ./ area(index(1:last-1));

        index([last; find(o>nms_thred)]) = [];
    end

    chosen = chosen(1:(count-1));
    chosen_detection = bbox_prob(chosen, :);

end