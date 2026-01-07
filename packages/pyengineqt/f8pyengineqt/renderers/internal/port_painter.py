from qtpy import QtCore, QtGui

def draw_nothing(painter, rect, info):
    pass

def draw_exec_port(painter, rect, info):
    """
    Custom paint function for drawing a Pentagon (UE5 Exec) shaped port.
    """
    painter.save()

    # Define dimensions relative to the rect
    w = rect.width()
    h = rect.height()
    half_w = w / 2
    half_h = h / 2
    
    # Create the pentagon (Exec pin shape)
    # Points are defined relative to (0,0) then transformed to rect center
    pentagon = QtGui.QPolygonF()
    
    # Left-Top
    pentagon.append(QtCore.QPointF(-half_w, -half_h))
    # Mid-Top (where the slant starts)
    pentagon.append(QtCore.QPointF(0.0, -half_h))
    # Tip (Right-Middle)
    pentagon.append(QtCore.QPointF(half_w, 0.0))
    # Mid-Bottom
    pentagon.append(QtCore.QPointF(0.0, half_h))
    # Left-Bottom
    pentagon.append(QtCore.QPointF(-half_w, half_h))

    # Map polygon to port position
    transform = QtGui.QTransform()
    transform.translate(rect.center().x(), rect.center().y())
    port_poly = transform.map(pentagon)

    # Styling logic
    if info['hovered']:
        color = QtGui.QColor(14, 45, 59)
        border_color = QtGui.QColor(136, 255, 35)
    elif info['connected']:
        # UE5 Exec pins are typically white when connected
        color = QtGui.QColor(*info['color'])
        border_color = QtGui.QColor(255, 255, 255)
    else:
        color = QtGui.QColor(0, 0, 0, 0) # Transparent fill if disconnected
        border_color = QtGui.QColor(*info['border_color'])

    pen = QtGui.QPen(border_color, 1.5)
    # MiterJoin ensures the 'tip' of the execution pin stays sharp
    pen.setJoinStyle(QtCore.Qt.MiterJoin)

    painter.setRenderHint(QtGui.QPainter.Antialiasing)
    painter.setPen(pen)
    painter.setBrush(color)
    painter.drawPolygon(port_poly)

    painter.restore()

def draw_signal_port(painter, rect, info):
    """
    Custom paint function for drawing a Signal/Delegate shaped port.
    (A circle with a small protruding arrow on the right).
    """
    painter.save()
    painter.setRenderHint(QtGui.QPainter.Antialiasing)

    # Calculate dimensions
    # We make the circle slightly smaller to leave room for the arrow tip
    radius = rect.height() * 0.35
    center = rect.center()
    
    # 1. Create the Circle part
    path = QtGui.QPainterPath()
    path.addEllipse(center, radius, radius)

    # 2. Create the Triangle (Arrow) part
    # We position it so it overlaps the right side of the circle
    tip_x = center.x() + radius*1.5
    base_x = center.x() + radius * 1.2
    offset_y = radius * 0.3
    
    arrow = QtGui.QPolygonF([
        QtCore.QPointF(base_x, center.y() - offset_y), # Top base
        QtCore.QPointF(tip_x, center.y()),              # Tip
        QtCore.QPointF(base_x, center.y() + offset_y)  # Bottom base
    ])
    
    # 3. Merge them
    # path.addPolygon effectively combines the shapes
    path.addPolygon(arrow)
    # Using simplified() removes the internal overlapping lines
    final_path = path.simplified()

    # Styling logic
    if info['hovered']:
        color = QtGui.QColor(14, 45, 59)
        border_color = QtGui.QColor(136, 255, 35)
    elif info['connected']:
        # Signals in UE are often a bright red or blue depending on context
        color = QtGui.QColor(*info['color'])
        border_color = QtGui.QColor(255, 255, 255)
    else:
        color = QtGui.QColor(0, 0, 0, 0)
        border_color = QtGui.QColor(*info['border_color'])

    pen = QtGui.QPen(border_color, 1.5)
    pen.setJoinStyle(QtCore.Qt.RoundJoin) # RoundJoin looks better on circular icons

    painter.setPen(pen)
    painter.setBrush(color)
    painter.drawPath(final_path)

    painter.restore()

def draw_triangle_port(painter, rect, info):
    """
    Custom paint function for drawing a Triangle shaped port.

    Args:
        painter (QtGui.QPainter): painter object.
        rect (QtCore.QRectF): port rect used to describe parameters needed to draw.
        info (dict): information describing the ports current state.
            {
                'port_type': 'in',
                'color': (0, 0, 0),
                'border_color': (255, 255, 255),
                'multi_connection': False,
                'connected': False,
                'hovered': False,
            }
    """
    painter.save()

    # create triangle polygon.
    size = int(rect.height() / 2)
    triangle = QtGui.QPolygonF()
    triangle.append(QtCore.QPointF(-size, size))
    triangle.append(QtCore.QPointF(0.0, -size))
    triangle.append(QtCore.QPointF(size, size))

    # map polygon to port position.
    transform = QtGui.QTransform()
    transform.translate(rect.center().x(), rect.center().y())
    port_poly = transform.map(triangle)

    # mouse over port color.
    if info['hovered']:
        color = QtGui.QColor(14, 45, 59)
        border_color = QtGui.QColor(136, 255, 35)
    # port connected color.
    elif info['connected']:
        color = QtGui.QColor(195, 60, 60)
        border_color = QtGui.QColor(200, 130, 70)
    # default port color
    else:
        color = QtGui.QColor(*info['color'])
        border_color = QtGui.QColor(*info['border_color'])

    pen = QtGui.QPen(border_color, 1.8)
    pen.setJoinStyle(QtCore.Qt.MiterJoin)

    painter.setPen(pen)
    painter.setBrush(color)
    painter.drawPolygon(port_poly)

    painter.restore()

def draw_square_port(painter, rect, info):
    """
    Custom paint function for drawing a Square shaped port.

    Args:
        painter (QtGui.QPainter): painter object.
        rect (QtCore.QRectF): port rect used to describe parameters needed to draw.
        info (dict): information describing the ports current state.
            {
                'port_type': 'in',
                'color': (0, 0, 0),
                'border_color': (255, 255, 255),
                'multi_connection': False,
                'connected': False,
                'hovered': False,
            }
    """
    painter.save()

    # mouse over port color.
    if info['hovered']:
        color = QtGui.QColor(14, 45, 59)
        border_color = QtGui.QColor(136, 255, 35, 255)
    # port connected color.
    elif info['connected']:
        color = QtGui.QColor(195, 60, 60)
        border_color = QtGui.QColor(200, 130, 70)
    # default port color
    else:
        color = QtGui.QColor(*info['color'])
        border_color = QtGui.QColor(*info['border_color'])

    pen = QtGui.QPen(border_color, 1.8)
    pen.setJoinStyle(QtCore.Qt.MiterJoin)

    painter.setPen(pen)
    painter.setBrush(color)
    painter.drawRect(rect)

    painter.restore()