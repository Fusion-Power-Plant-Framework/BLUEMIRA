# 5. Caching vs. recreation of geometry from Matryoshkas

Date: 2022-06-15 - 2022-11-30

## Status

Completed

## Context

At present we recreate FreeCAD geometries by travelling through the boundaries of their
"Matryoshka". We have a number of issues relating to this as our implementation appears
 to be imperfect.

Assumption: `BluemiraGeo` objects are "static".

Options:
* Continue with recreating the shapes
    * Pros:
        * We can track labels throughout the creation of geometry
        * Meshing and FE problems are easier to set up because of the above
    * Cons:
        * Higher computational cost (recreate upon each call of geometry)
        * Higher maintenance overhead
        * Reverse engineering of FreeCAD is presently imperfect
            * An operation can work in FreeCAD but fail upon "reconstruction" in bluemira
            * NotClosedWireError, DisjointedFaceError, ...
* Cache the FreeCAD shape
    * Pros:
        * Lower computational cost
        * Lower maintenance cost
        * Less logic and code
    * Cons:
        * If we want to label specific boundaries of geometry, we need to travel through
        all the boundaries and find the "right" ones and label post-instantiation (fiddly!)
        * This means that we cannot track shared boundaries (upon creation, but see above)
        * Meshing and FE problems are more complicated to set up
        * Recreation of geometry is not totally avoided, however, as there is
        still a need to serialize/deserialize objects.

## Decision

The recreation of the shapes was causing serious, frequent, often not-perfectly-reproducible
problems when trying to run the code. After experimenting with geometry caching, noting that
it was more stable and faster, we decided to cache the underlying FreeCAD shapes, rather than
recreate them.

## Consequences

The known consequences are, generally speaking, that setting up finite element problems is going
to be trickier, particularly when it comes to finding the right boundaries post-creation.

At present, only BluemiraWires support serialisation/deserialise, minimising the amount of recreation involved in the present implementation. Longer-term, though, this is likely to cause problems.

## Additional context

It is possible that future changes, either in terms of moving from FreeCAD to
another library, or in terms of FreeCAD's scheduled v1 release, will change
the landscape of this decision, and we will need to re-assess.
