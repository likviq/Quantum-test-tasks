from collections import deque

def is_new_island(map_details, row_index, column_index):
    """
    Check if the given cell is part of a new island.

    An island is defined as a cell with a value of 1 that has not been visited yet.

    Args:
        map_details (list[list[int]]): The map represented as a 2D list of integers.
        row_index (int): The row index of the cell to check.
        column_index (int): The column index of the cell to check.

    Returns:
        bool: True if the cell is part of a new island, False otherwise.
    """
    map_state = map_details[row_index][column_index]
    return map_state == 1

def add_neighbor_nodes(nodes, map_details, row_index, column_index):
    """
    Add neighboring cells of the current cell to the queue if they are part of the island.

    Args:
        nodes (deque): Queue of cells to process.
        map_details (list[list[int]]): The map represented as a 2D list of integers.
        row_index (int): The row index of the current cell.
        column_index (int): The column index of the current cell.

    Returns:
        deque: Updated queue with valid neighboring cells added.
    """
    rows_amount, columns_amount = len(map_details), len(map_details[0])

    if row_index > 0 and map_details[row_index - 1][column_index] == 1:
        nodes.append((row_index - 1, column_index))
    if column_index + 1 < columns_amount and map_details[row_index][column_index + 1] == 1:
        nodes.append((row_index, column_index + 1))
    if row_index + 1 < rows_amount and map_details[row_index + 1][column_index] == 1:
        nodes.append((row_index + 1, column_index))
    if column_index > 0 and map_details[row_index][column_index - 1] == 1:
        nodes.append((row_index, column_index - 1))

    return nodes

def remove_island_from_map(map_details, row_index, column_index):
    """
    Remove all cells belonging to an island starting from the given cell.

    This function modifies the map in place, marking all cells of the island as visited (value 0).

    Args:
        map_details (list[list[int]]): The map represented as a 2D list of integers.
        row_index (int): The starting row index of the island.
        column_index (int): The starting column index of the island.

    Returns:
        list[list[int]]: The updated map with the island removed.
    """
    nodes = deque([(row_index, column_index)])
    
    while nodes:
        row_index, column_index = nodes.popleft()
        map_details[row_index][column_index] = 0
        nodes = add_neighbor_nodes(nodes, map_details, row_index, column_index)
    
    return map_details

def main():
    """
    Main function to count the number of islands in a map.

    Reads the map dimensions and values from the user, processes the map, and prints the number of islands found.
    """
    rows, columns = map(int, input().split())
    map_details = [list(map(int, input().split())) for _ in range(rows)]

    islands_amount = 0
    for row_index in range(rows):
        for column_index in range(columns):
            if not is_new_island(map_details, row_index, column_index):
                continue
            islands_amount += 1
            map_details = remove_island_from_map(map_details, row_index, column_index)

    print(islands_amount)

if __name__ == "__main__":
    main()
