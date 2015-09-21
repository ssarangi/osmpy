# Shortest Path on OSM files
Shortest path algorithms by loading OSM files and then clicking on two points to calculate the shortest paths.
The snapping of points on matplotlib image isn't very accurate causing the nodes to snap to their nearest location. That
is why you would see that the start / end points might not match the actual path shown.

![Alt text](/shortest_path.png?raw=true "Shortest Path Algorithms")

Can be invoked with -
For Matplotlib
python osm_to_networkx.py data\smaller.osm

For OpenGL / VisPy
python osm_to_networkx.py data\smaller.osm renderer

![Alt text](/vispy_rendering.png?raw=true "Vispy rendering")