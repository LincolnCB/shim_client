"""
Interactive plot utility functions for hover and click interactions.
"""
import numpy as np


def on_hover(event, ax, lines, annotation, highlighted_line, label_prefix="Channel"):
    """
    Handle hover events over plotted lines.
    
    Parameters:
    -----------
    event : matplotlib event
        The mouse event
    ax : matplotlib axes
        The axes containing the lines
    lines : list
        List of line objects to check for hover
    annotation : matplotlib annotation
        The annotation object to update
    highlighted_line : list
        List containing the currently highlighted line (mutable reference)
    label_prefix : str
        Prefix for the annotation label (e.g., "Channel", "Slice")
    """
    if event.inaxes != ax:
        # Hide annotation if mouse leaves the plot
        annotation.set_visible(False)
        if highlighted_line[0] is not None:
            highlighted_line[0].set_linewidth(2)
            highlighted_line[0].set_alpha(0.5)
            highlighted_line[0] = None
        event.canvas.figure.canvas.draw_idle()
        return
    
    # Check if hovering over any line
    found = False
    for i, line in enumerate(lines):
        if line.contains(event)[0]:
            found = True
            # Reset previous highlight
            if highlighted_line[0] is not None and highlighted_line[0] != line:
                highlighted_line[0].set_linewidth(2)
                highlighted_line[0].set_alpha(0.5)
            
            # Highlight current line
            line.set_linewidth(3)
            line.set_alpha(1.0)
            highlighted_line[0] = line
            
            # Update annotation
            annotation.set_text(f'{label_prefix} {i}')
            annotation.xy = (event.xdata, event.ydata)
            annotation.set_visible(True)
            event.canvas.figure.canvas.draw_idle()
            break
    
    if not found:
        # No line under cursor
        annotation.set_visible(False)
        if highlighted_line[0] is not None:
            highlighted_line[0].set_linewidth(2)
            highlighted_line[0].set_alpha(0.5)
            highlighted_line[0] = None
        event.canvas.figure.canvas.draw_idle()


def on_click(event, ax, lines, original_ylim=None):
    """
    Handle click events to set y-axis limits based on clicked line.
    
    Parameters:
    -----------
    event : matplotlib event
        The mouse event
    ax : matplotlib axes
        The axes containing the lines
    lines : list
        List of line objects to check for clicks
    original_ylim : tuple, optional
        Original y-axis limits to restore when clicking off lines
    """
    if event.inaxes != ax:
        return
    
    # Check if clicking on any line
    clicked_line = False
    for line in lines:
        if line.contains(event)[0]:
            clicked_line = True
            # Get the y-data of the clicked line
            ydata = line.get_ydata()
            ymin = np.min(ydata)
            ymax = np.max(ydata)
            
            # Add some padding (5% of range)
            y_range = ymax - ymin
            padding = y_range * 0.05
            
            # Set y-axis limits
            ax.set_ylim(ymin - padding, ymax + padding)
            event.canvas.figure.canvas.draw_idle()
            break
    
    # If clicked off all lines, reset to original limits
    if not clicked_line:
        if original_ylim is not None:
            ax.set_ylim(original_ylim)
        else:
            # Calculate limits from all lines
            all_ydata = np.concatenate([line.get_ydata() for line in lines])
            ymin, ymax = np.min(all_ydata), np.max(all_ydata)
            y_range = ymax - ymin
            padding = y_range * 0.05
            ax.set_ylim(ymin - padding, ymax + padding)
        event.canvas.figure.canvas.draw_idle()
