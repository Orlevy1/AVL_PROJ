# username - nadavlevi,orlevy
# id1      - 314831017
# name1    - nadav levi
# id2      - 313354508
# name2    - or levy

"""A class represnting a node in an AVL tree"""
import random


class AVLNode(object):
    """Constructor, you are allowed to add more fields.

    @type value: str
    @param value: data of your node
    """

    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        self.parent = None
        self.height = -1  # Balance factor
        self.size = 0



    """returns the left child
    @rtype: AVLNode
    @returns: the left child of self, None if there is no left child
    """

    def getLeft(self):
        return self.left

    """returns the right child

    @rtype: AVLNode
    @returns: the right child of self, None if there is no right child
    """

    def getRight(self):
        return self.right

    """returns the parent 

    @rtype: AVLNode
    @returns: the parent of self, None if there is no parent
    """

    def getParent(self):
        return self.parent

    """return the value

    @rtype: str
    @returns: the value of self, None if the node is virtual
    """

    def getValue(self):
        return self.value

    """returns the height

    @rtype: int
    @returns: the height of self, -1 if the node is virtual
    """

    def getHeight(self):
        return self.height


    """returns the size

    @rtype:int
     @return the size of self, o if node is virtual
    """


    def getSize(self):
        return self.size

    """returns the balance factor of the node
    
    @rtype:int
    returns: balance factor"""
    def getBalanceFactor(self):
        return self.left.getHeight() - self.right.getHeight()

    """sets left child
    
      @type node: AVLNode
      @param node: a node
      """


    def setLeft(self, node):
        self.left = node


    """sets right child
    
    @type node: AVLNode
    @param node: a node
    """


    def setRight(self, node):
        self.right = node


    """sets parent
    
    @type node: AVLNode
    @param node: a node
    """


    def setParent(self, node):
        self.parent = node


    """sets value
    
    @type value: str
    @param value: data
    """


    def setValue(self, value):
        self.value = value


    """sets the balance factor of the node
    
    @type h: int
    @param h: the height
    """


    def setHeight(self, h):
        self.height = h;


    """sets the size of the node
    
    @type s: int
    @param s: the size
    """


    def setSize(self, s):
        self.size = s;


    """returns whether self is not a virtual node 
    
    @rtype: bool
    @returns: False if self is a virtual node, True otherwise.
    """


    def isRealNode(self):
        return self.size > 0 and self.height >= 0

    """updates the size and height of the node
    
    @rtype = None
    """
    def updateSizeHeight(self):
        self.size = 1 + self.right.size + self.left.size
        self.height = 1 + max(self.right.getHeight(),self.left.getHeight())
        if not self.left.isRealNode() and not self.right.isRealNode:
            self.height = 0


    """init an new leaf
    @rtype:None"""
    def buildLeaf(self):
        self.size = 1
        self.height=0
        self.right = AVLNode(None)
        self.left = AVLNode(None)
        self.right.setParent(self)
        self.left.setParent(self)
    """
    A class implementing the ADT list, using an AVL tree.
    """


class AVLTreeList(object):
    """
    Constructor, you are allowed to add more fields.

    """

    def __init__(self):
        self.size = 0
        self.root = None
        self.firstItem = None
        self.lastItem = None

    # add your fields here

    """returns whether the list is empty

    @rtype: bool
    @returns: True if the list is empty, False otherwise
    """

    def empty(self):
        if self.root == None or self.root.value == None:
            return True
        return False

    """retrieves the value of the i'th item in the list

    @type i: int
    @pre: 0 <= i < self.length()
    @param i: index in the list
    @rtype: str
    @returns: the the value of the i'th item in the list
    @Complexity: O(log(n)) - Select cost is log(n)
    """

    def retrieve(self, i):
        if self.getRoot() is None:
            return None
        return self.Select(self.root,i+1).getValue()

    """inserts val at position i in the list

    @type i: int
    @pre: 0 <= i <= self.length()
    @param i: The intended index in the list to which we insert val
    @type val: str
    @param val: the value we inserts
    @rtype: list
    @returns: the number of rebalancing operation due to AVL rebalancing
    @complexity: O(log(n)) - cost of selecting next node, finding predeseccor,fixing height,size,BF is all log(n)
    """

    def insert(self, i, val):
        newNode = AVLNode(val)
        newNode.buildLeaf()
        if self.size == 0:  #inserts new minimum
            self.root = newNode
            self.firstItem = self.root
            self.lastItem = self.root
            self.root.getRight().setParent(self.root)
            self.root.getLeft().setParent(self.root)


        elif i == self.size: #inserts new maximum
            maxi = self.maximum(self.root)
            maxi.setRight(newNode)
            newNode.setParent(maxi)
            newNode.buildLeaf()
            self.lastItem = newNode


        elif i < self.size:
            next_node =self.Select(self.root,i+1)
            if not next_node.getLeft().isRealNode(): #dosent have left child
                next_node.setLeft(newNode)
                newNode.setParent(next_node)
                newNode.buildLeaf()
                if i == 0:
                    self.firstItem = next_node.getLeft()

            else:  #have left child
                prede = self.predecessor(next_node)
                prede.setRight(newNode)
                newNode.setParent(prede)
                newNode.buildLeaf()

                if i == 0:
                    self.firstItem = prede.getRight()

        self.size += 1
        x = self.fixHeightSizeInsertBalance(newNode)
        self.fixHeightSizeInsert(newNode)
        return x




    """deletes the i'th item in the list

    @type i: int
    @pre: 0 <= i < self.length()
    @param i: The intended index in the list to be deleted
    @rtype: int
    @returns: the number of rebalancing operation due to AVL rebalancing
    @Complexity: cost of selecting the Node, Find successor, and fixing the tree is all log(n)
    """

    def delete(self, i): # delete in O(logn)

        DeleteNode = self.Select(self.root,i+1) # node is the one that we want to delete from the tree

        if DeleteNode.height == 0:  # the node is a leaf
            self.deleteLeaf(DeleteNode)
            parent = DeleteNode.getParent()
        elif (DeleteNode.getRight().size == 0 or DeleteNode.getLeft().size == 0):  # the node has only one child
            self.deleteOneChild(DeleteNode)
            parent = DeleteNode.getParent()
        else:  # the node has two children
            successor = self.Select(self.root,i+2)  # successor of node i
            DeleteNode.setValue(successor.value)
            DeleteNode = successor # now we want actually to delete the successor of the node
            parent = DeleteNode.getParent()
            self.deleteTwoChildren(successor)

        self.size -= 1
        self.fixHeightSizeDelete(parent)
        self.updateFandLafterDelete(i)
        x = self.fixBalanceFactorDelete(parent)

        #check mistakes for extreme cases, in normal case will take O(1)
        self.fixHeightSizeDelete(parent)
        if self.root != None and abs(self.root.getBalanceFactor()) > 1:
            x += self.fixBalanceFactorDelete(parent)
        return x


    """help function for delete for case of deleting  leaf"""
    def deleteLeaf(self, node):
        if node == self.root:  # the tree has one node, after the deletion he will be empty
            self.root = None
            node.getRight().setParent(self)
            node.getLeft().setParent(self)
        elif (node.getParent().getRight() == node):  # the node is a right child
            node.getParent().setRight(node.getRight())
            node.getParent().getRight().setParent(node.getParent())
        else:  # node is a left child
            node.getParent().setLeft(node.getRight())
            node.getRight().setParent(node.getParent())
        return

    """help function for delete for case of deleting a node with exactly one child"""
    def deleteOneChild(self, node):  # delete a node that has only one child in O(1)
        if node == self.root:  # the root is the node we want to delete
            if node.getRight().size == 1:  # node is the root and has a one right child
                self.root = node.getRight()
                node.getRight().setParent(None)
            elif node.getLeft().size == 1:  # node is the root and has a one left child
                self.root = node.getLeft()
                node.getLeft().setParent(None)
        else:
            if node.getRight().getSize() == 1:  # node has a one right child
                curr = node.getRight()
                curr.setParent(node.getParent())
                if node.getParent().getLeft() == node:  # node is a left child
                    node.getParent().setLeft(curr)
                else:  # node is a right child
                    node.getParent().setRight(curr)
            else:  # node has a one left child
                curr = node.getLeft()
                curr.setParent(node.getParent())
                if node.getParent().getLeft() == node:  # node is a left child
                    node.getParent().setLeft(curr)
                else:  # node is a right child
                    node.getParent().setRight(curr)

    """help function for delete for case of deleting for a node with two childrens"""
    def deleteTwoChildren(self, DeleteNode):  # delete a node that has two children in O(logn)
        if (DeleteNode.getHeight() == 0):  # successor is a leaf
            self.deleteLeaf(DeleteNode)
        elif self.root == DeleteNode:  # we want to delete the root of the tree
            if DeleteNode.getLeft().isRealNode():
                DeleteNode.getLeft().setParent(DeleteNode.getParent())
                self.root = DeleteNode.getLeft()
            else:
                DeleteNode.getRight().setParent(DeleteNode.getParent())
                self.root = DeleteNode.getRight()
        elif DeleteNode.getParent().getLeft() == DeleteNode:  # the DeleteNode is a left child
            if DeleteNode.getLeft().isRealNode():  # the DeleteNode has left child
                DeleteNode.getParent().setLeft(DeleteNode.getLeft())
                DeleteNode.getLeft().setParent(DeleteNode.getParent())
            else:
                DeleteNode.getParent().setLeft(DeleteNode.getRight())
                DeleteNode.getRight().setParent(DeleteNode.getParent())
        elif DeleteNode.getParent().getRight() == DeleteNode:  # the DeleteNode is a right child
            if DeleteNode.getLeft().isRealNode():  # the DeleteNode has left child
                DeleteNode.getParent().setRight(DeleteNode.getLeft())
                DeleteNode.getLeft().setParent(DeleteNode.getParent())
            else:
                DeleteNode.getParent().setRight(DeleteNode.getRight())
                DeleteNode.getRight().setParent(DeleteNode.getParent())
        return 0

    """function that update the value of first and last after delete action only if necessary in O(logn)"""
    def updateFandLafterDelete(self,i):
        if self.empty(): # list is empty
            self.firstItem = None
            self.lastItem = None
        else: #list isn't empty
            if i == self.length():
                self.lastItem = self.Select(self.root,self.length())
            if i == 0:
                self.firstItem = self.Select(self.root,1)


    """returns the value of the first item in the list

    @rtype: str
    @returns: the value of the first item, None if the list is empty
    @complexity: O(1) - returning a variable is O(1)
    """

    def first(self):
        if self.firstItem == None or self.firstItem.value == None:
            return None
        return self.firstItem.value

    """returns the value of the last item in the list

    @rtype: str
    @returns: the value of the last item, None if the list is empty
    @complexity: O(1) - returning a variable is O(1)
    """

    def last(self):
        if self.lastItem == None or self.firstItem.value == None:
            return None
        return self.lastItem.value

    """returns an array representing list 

    @rtype: list
    @returns: a list of strings representing the data structure
    @Complexity: same as in Order scan O(n)
    """

    def listToArray(self):
        if (self.root == None):
            return []
        def listToArrayRec(node):
            if not node.isRealNode():
                return []
            return listToArrayRec(node.getLeft()) + [node.getValue()] + listToArrayRec(node.getRight())
        return listToArrayRec(self.root)

    """returns the size of the list 

    @rtype: int
    @returns: the size of the list
    """

    def length(self):
        return self.size

    """sort the info values of the list

    @rtype: list
    @returns: an AVLTreeList where the values are sorted by the info of the original list.
    @complexity: mergesort cost O(nlogn) n insert in max cost of log(n) each is O(nlog(n)) total
    """

    def sort(self):
        array = self.mergesort(self.listToArray())

        sorted_tree = AVLTreeList()
        for i in range(len(array)):
            sorted_tree.insert(i,array[i])
        return sorted_tree
    """permute the info values of the list 

    @rtype: list
    @returns: an AVLTreeList where the values are permuted randomly by the info of the original list. ##Use Randomness
    @Complexity;
    """

    def permutation(self):
        array = self.listToArray()
        for i in range(len(array)): #shuffels the list using random
            j = round(random.random()*(len(array)-1))
            temp = array[i]
            array[i] = array[j]
            array[j] = temp

        """a recursive function that runs on the random list and builds a tree from it in O(n)"""
        def buildTreeRec(list,left,right,node):
            if left == right:
                leftnode = AVLNode(list[left])
                leftnode.buildLeaf()
                node.setLeft(leftnode)
                node.getLeft().setParent(node)
                node.setSize(2)
                node.setHeight(1)
                return
            if left>right: return
            leftnode = AVLNode(list[left])
            leftnode.buildLeaf()
            node.setLeft(leftnode)
            rightnode = AVLNode(list[right])
            rightnode.buildLeaf()
            node.setRight(rightnode)
            node.getLeft().setParent(node)
            node.getRight().setParent(node)
            buildTreeRec(list,left+1,right//2,node.getLeft())
            if not (right//2 <= left):
                buildTreeRec(list, right//2 + 1, right-1, node.getRight())
            node.updateSizeHeight()

        #init the new rand_tree and fixes
        rand_tree = AVLTreeList()
        rand_tree.root = AVLNode(array[0])
        buildTreeRec(array,1,len(array)-1,rand_tree.root)
        rand_tree.root.updateSizeHeight()
        rand_tree.size = rand_tree.root.getLeft().getSize() +rand_tree.root.getRight().getSize() +1
        rand_tree.lastItem = self.maximum(rand_tree.root)
        rand_tree.firstItem = self.minimum(rand_tree.root)
        return rand_tree




    """concatenates lst to self

    @type lst: AVLTreeList
    @param lst: a list to be concatenated after self
    @rtype: int
    @returns: the absolute value of the difference between the height of the AVL trees joined
    """

    def concat(self, lst):
        if lst.root is None and self.root is None: #both empty lists
            return 0
        if(lst.root is None): #second list is empty list so no concat needed
            dif = abs(self.root.getHeight())
            return dif
        elif self.root is None: #first list is empty list
            dif = abs(lst.getRoot().getHeight())
            self.root = lst.root
            self.firstItem=lst.firstItem
            self.lastItem=lst.lastItem
            self.size=lst.size
            return dif
        else: #normal case
            dif = abs(self.root.getHeight() - lst.getRoot().getHeight())
        if(lst.root is not None and self.root.getHeight() <= lst.getRoot().getHeight()):
            #second list lst is higher then self concats self on the left side of lst
            if self.getRoot().getSize() > 1:
                x = self.lastItem
                self.delete(self.size-1)
                x.setLeft(self.root)
                x.getLeft().setParent(x)
            else:
                x = self.root
            b = lst.getRoot()
            if (b.getHeight() == self.root.getHeight()):
                b=b.getLeft()
            while b.getHeight() > self.root.getHeight():
                b= b.getLeft()
            x.setRight(b)
            c = b.getParent()
            b.setParent(x)
            c.setLeft(x)
            x.setParent(c)
            self.root = lst.root
            self.size = lst.size + self.size + 1
            self.lastItem = lst.lastItem
            self.fixHeightSizeConcat(x)
            self.fixBalanceFactorDelete(x)
            self.fixHeightSizeConcat(x)
            return dif


        else:#first list (self) is higher then lst , concats lst in the right side of self
            if(lst.root.getSize() > 1):
                x = lst.firstItem
                lst.delete(lst.size-1)
                x.setRight(lst.root)
            else:
                return dif
            if(x.getRight() != None):
                x.getRight().setParent(x)
            b = self.getRoot()
            if (b.getHeight() == lst.root.getHeight()):
                b=b.getRight()
            while b.getHeight() > lst.root.getHeight():
                b= b.getRight()
            x.setLeft(b)
            c = b.getParent()
            b.setParent(x)
            c.setRight(x)
            x.setParent(c)

            self.size = lst.size + self.size + 1
            self.lastItem = lst.lastItem
            self.fixHeightSizeConcat(x)
            self.fixBalanceFactorDelete(x)
            self.fixHeightSizeConcat(x)
            return dif

    """Fixes the height and size of the concatinated tree"""
    def fixHeightSizeConcat(self,node):

        y = node
        while y != None:
            y.setSize(1+y.getLeft().getSize()+y.getRight().getSize())
            y.setHeight(1+max(y.getLeft().getHeight(),y.getRight().getHeight()))
            y = y.getParent()
    """searches for a *value* in the list

    @type val: str
    @param val: a value to be searched
    @rtype: int
    @returns: the first index that contains val, -1 if not found.
    """

    def search(self, val):
        lst = self.listToArray()
        for i in range(len(lst)):
            if lst[i] == val:
                return i
        return -1


    """returns the root of the tree representing the list

    @rtype: AVLNode
    @returns: the root, None if the list is empty
    """

    def getRoot(self):
        return self.root

    """return the 𝒌th smallest element in T
    
    @rtype = AVLNode
    @returns: 𝒌th smallest element, None if k<1 or k>root.size
    @Complexity: maximum height of the tree actions in AVL meaning O(log(n))"""
    def Select(self,node,k):
        if not node.isRealNode():
          return node

        r = node.left.getSize() + 1
        if k == r: return node
        elif k < r: return self.Select(node.getLeft(),k)
        elif k > r: return self.Select(node.getRight(),k-r)


    """return the predecessor of a given node
    
    @rtype = AVLNode
    @returns: predecessor
    @Complexity: maximum height of the tree actions in AVL meaning O(log(n))"""
    def predecessor(self,node):
        if node.getLeft().isRealNode():
            return self.maximum(node.getLeft())
        y = node.getParent()
        while(y.isRealNode() and node == y.getLeft()):
            node = y
            y = node.getParent
        return y

    """checking for invalid balance factor after insertion and corrects it
    
    @rtype = int
    @returns:number of rebalncing opertatins
    @Complexity: O(log(n)) maximum height of tree action till finding invalid balance factor and O(1) for rotate"""
    def fixHeightSizeInsertBalance(self,node):
        y = node.getParent()
        while(y is not None):
            bf = y.getBalanceFactor()
            if (abs(bf) < 2 and y.getHeight() == (max(y.getRight().getHeight(),y.getLeft().getHeight())+1)):#height hasnt changed
                return 0
            elif abs(bf) < 2 and y.getHeight() != (max(y.getRight().getHeight(),y.getLeft().getHeight())+1):#height has changed,update it
                y.setHeight(max(y.getRight().getHeight(), y.getLeft().getHeight()) + 1)
                y.setSize(y.getLeft().getSize() + y.getRight().getSize() + 1)
                y = y.getParent()
            elif abs(bf) > 1:
                return self.rotateNode(y)
        return 0

    """"checking for invalid balance factor after deletion and corrects it
    
    @rtype = int
    @returns:number of rebalancing operations
    @Complexity: O(log(n)) height of tree actions and O(1) for each rotate"""
    def fixBalanceFactorDelete(self,node):
        y = node
        count =0
        while (y is not None):
            bf = y.getBalanceFactor()
            if(abs(bf)>1):
                count += self.rotateNode(y)
            y = y.getParent()

        return count
    """rotates an unbalanced node
    
    @rtype:int
    @returns:1 if one rotation 2 if double rotation
    Complexity: O(1)"""
    def rotateNode(self,node):

        bf = node.getBalanceFactor()
        if (bf == 2):
            son = node.getLeft()
            sonBF = son.getBalanceFactor()
            if(sonBF == 1):
                self.rightRotation(node)
                return 1
            if (sonBF == 0):
                self.rightRotation(node)
                return 1
            elif(sonBF == -1):
                self.leftRotation(son)
                self.rightRotation(node)
                return 2
        elif(bf == -2):
            son = node.getRight()
            sonBf = son.getBalanceFactor()
            if(sonBf == -1):
                self.leftRotation(node)
                return 1
            if (sonBf == 0):
                self.leftRotation(node)
                return 1
            elif(sonBf == 1):
                self.rightRotation(son)
                self.leftRotation(node)
                return 2
        return 0

    """fixes size and heights of path to the root after insertion
    
    @rtype:None
    @return:None
    @complexity: O(log(n)) path from node to root is max tree height"""

    def fixHeightSizeInsert(self,node):
        y = node
        while(y is not None):
            new_height = max(y.getRight().getHeight(), y.getLeft().getHeight()) + 1
            new_size = y.getLeft().getSize() + y.getRight().getSize() + 1
            if (y.getSize() == new_size and y.getHeight == new_height):
                return
            y.setHeight(new_height)
            y.setSize(new_size)
            y = y.getParent()

    """fixes size and heights of path to the root after deletion

    @rtype:None
    @return:None
    @complexity: O(log(n)) path from node to root is max tree height"""

    def fixHeightSizeDelete(self,node):
        y = node
        while(y is not None):
            new_height = max(y.getRight().getHeight(),y.getLeft().getHeight())+1
            new_size = y.getLeft().getSize()+y.getRight().getSize()+1
            if (y.getSize() == new_size and y.getHeight == new_height):
                return
            y.setHeight(new_height)
            y.setSize(new_size)
            y= y.getParent()

    """rotates the node to the right side maintaining size and height
    
    @rtype:None
    @return:None
    @complexity:O(1)"""
    def rightRotation(self,B):
        A = B.getLeft()
        B.setLeft(A.getRight())
        B.getLeft().setParent(B)
        A.setRight(B)
        A.setParent(B.getParent())
        if(B.getParent() == None): self.root = A
        elif(B.getParent().getRight() == B): A.getParent().setRight(A)
        else: A.getParent().setLeft(A)
        B.setParent(A)
        B.updateSizeHeight()
        A.updateSizeHeight()
        B.updateSizeHeight()
        A.updateSizeHeight()


    """rotates the node to the left side maintaining size and height

    @rtype:None
    @return:None
    @complexity:O(1)"""
    def leftRotation(self,B):
        A = B.getRight()
        B.setRight(A.getLeft())
        B.getRight().setParent(B)
        A.setLeft(B)
        A.setParent(B.getParent())
        if (B.getParent() == None):
            self.root = A
        elif (B.getParent().getRight() == B):
            A.getParent().setRight(A)
        else:
            A.getParent().setLeft(A)
        B.setParent(A)
        B.updateSizeHeight()
        A.updateSizeHeight()


    """"returns the maximum/minimum element (by tree defenition) starting from specific node 
    
    @rtype = AVLNode
    @returns: maximum/minimum element
    complexity: O(log(n)) path from node to max in subtree is max tree height"""
    def maximum(self,node):
        y = node
        while(y.isRealNode()):
            node = y
            y=y.getRight()
        return node

    def minimum(self,node):
        y = node
        while(y.isRealNode()):
            node = y
            y=y.getLeft()
        return node

    """return the root Node
    
    @rtype:AVL Node
    @returns: the root"""
    def getRoot(self):
        return self.root



    """ merging two lists into a sorted list
            A and B must be sorted!
             
    @rtype: list
    returns: the merged list"""
    def merge(self,A, B):

        n = len(A)
        m = len(B)
        C = [None for i in range(n + m)]

        a = 0;
        b = 0;
        c = 0
        while a < n and b < m:  # more element in both A and B
            if A[a] < B[b]:
                C[c] = A[a]
                a += 1
            else:
                C[c] = B[b]
                b += 1
            c += 1

        C[c:] = A[a:] + B[b:]  # append remaining elements (one of those is empty)

        return C

    """ recursive mergesort sorting the incoming list using merge
     
     @rtype:list
     returns: the sorted list"""
    def mergesort(self,lst):

        n = len(lst)
        if n <= 1:
            return lst
        else:  # two recursive calls, then merge
            return self.merge(self.mergesort(lst[0:n // 2]),self.mergesort(lst[n // 2:n]))

