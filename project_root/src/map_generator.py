import folium

def generate_map(predictions, true_labels, map_file_path):
    m = folium.Map(location=[true_labels[0][0], true_labels[0][1]], zoom_start=12)

    for pred, true in zip(predictions, true_labels):
        folium.Marker(
            location=[true[0], true[1]],
            popup='True Location',
            icon=folium.Icon(color='green')
        ).add_to(m)

        folium.Marker(
            location=[pred[0], pred[1]],
            popup='Predicted Location',
            icon=folium.Icon(color='red')
        ).add_to(m)

    m.save(map_file_path)