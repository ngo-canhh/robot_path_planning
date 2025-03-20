  # Top-left rectangle (short)
        StaticObstacle(Rectangle(20, 5, 3, 5)),
        
        # Oval/Circle in the middle-left
        StaticObstacle(Circle(15, 30, 4)),
        
        # Triangle at top-right (approximated with a rectangle)
        StaticObstacle(Rectangle(35, 5, 7, 7)),
        
        # Cross/plus sign at top-right
        StaticObstacle(Rectangle(30, 25, 10, 2)),  # Horizontal part
        StaticObstacle(Rectangle(34, 21, 2, 10)) , # Vertical part        
        # Hexagon at bottom-left (approximated with a circle)
        StaticObstacle(Rectangle(8, 47, 6, 10)),
        
        # Pentagon at bottom-right (approximated with a circle)
        StaticObstacle(Circle(43, 45, 4))