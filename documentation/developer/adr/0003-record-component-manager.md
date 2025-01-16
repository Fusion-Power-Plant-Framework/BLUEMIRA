# 3. `ComponentManager`

Date: 2022-06-30
Updated: 2022-10-24

## Status

Completed

## Context

At present access to information relevant to a given `Component` is difficult / unintuitive
to retrieve. In many situations we want short-hand access to these characteristics / attributes / properties.

Options:
* `Component` is more "active" and directly has e.g. properties and methods to retrieve information
* `ComponentManager` is used to provide access to information relating to an abstract `Component`
    * At the `ReactorDesigner` level or `Component` level.
* `Builder` does everything (builds the `Component` and logic + access methods to extract information + store the `Component`)

It will be quite difficult to have a totally "general" `ComponentManager`.

Many of the properties are not specific to a `Component` but may require access to "external"
information present in the `Reactor` or other `Component`s.

## Decision

We will trial the three options above taking e.g. the EUDEMO plasma `Component` as an example
to inform our decision.

#### 2022-10-24

A `ComponentManager` was settled on as a good abstraction from a `Component`, physical or otherwise, and information that a user
may require from a `Component`. For example:

* The separatrix of a plasma is part of a component but requires low level knowledge to access
* The magnetic field due to the equlibria is interlinked with the plasma but not part of the `Component`
    * The equlibiria can be stored on the `ComponentManager`

## Consequences

There will be a few internal `ComponentManager`s available for use. It is however expected that a Reactor Designer creates
their own `ComponentManager`s to facilitate their workflow. `ComponentManager` objects will only live at the designer/builder level,
facilitating access to properties that can be passed to lower level objects.
`ComponentManagers` abstract away the component tree structure at the reactor level.
In theory, this means the component trees can be changed without having to make changes to the reactor build workflow.
