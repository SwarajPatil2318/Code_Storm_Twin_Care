/** Bridge for the P8 vanilla shell: the exact same React wearable as both Twins. */
import {createRoot, type Root} from 'react-dom/client';
import {WearableMiniCard} from './WearableMiniCard';
import type {ComponentProps} from 'react';
const roots=new WeakMap<HTMLElement,Root>();
export function mountWearable(target:HTMLElement,props:ComponentProps<typeof WearableMiniCard>) {
  let root=roots.get(target);if(!root){root=createRoot(target);roots.set(target,root);}
  root.render(<WearableMiniCard {...props}/>);
}
export function unmountWearable(target:HTMLElement){roots.get(target)?.unmount();roots.delete(target);}
