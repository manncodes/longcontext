def group_segments(self, text, max_length=2048):

    segments = text.split('\n')
    
    groups = []
    current_group = []
    current_length = 0
    

    for segment in segments:
        segment_length = len(segment)
        
        if current_length + segment_length > max_length:
            if current_length!=0:
                groups.append('\n'.join(current_group))
            current_group = [segment]
            current_length = segment_length

        else:

            current_group.append(segment)
            current_length += segment_length

    if current_group:

        groups.append('\n'.join(current_group))

    return groups