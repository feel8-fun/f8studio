import { useEffect } from 'react';
import { Link, useSearchParams } from 'react-router-dom';

export function AuthCallbackRoute() {
  const [searchParams] = useSearchParams();
  const status = String(searchParams.get('status') || '').trim().toLowerCase() || 'success';
  const isSuccess = status !== 'error';
  const error = String(searchParams.get('error') || '').trim();
  const errorDescription = String(searchParams.get('error_description') || '').trim();

  useEffect(() => {
    if (!isSuccess) {
      return undefined;
    }
    const timeoutId = window.setTimeout(() => {
      window.location.assign('/assets/mine');
    }, 2500);
    return () => {
      window.clearTimeout(timeoutId);
    };
  }, [isSuccess]);

  return (
    <div className="flex w-full items-center justify-center">
      <div className="w-full max-w-xl rounded-[2rem] border border-white/10 bg-slate-950/45 p-6 shadow-[0_30px_90px_rgba(0,0,0,0.32)] backdrop-blur-xl">
        <p className="text-xs uppercase tracking-[0.34em] text-cyan-200/70">Desktop Sign-In</p>
        <h2 className="mt-3 text-3xl font-semibold text-white">
          {isSuccess ? 'Desktop sign-in complete' : 'Desktop sign-in needs attention'}
        </h2>
        <p className="mt-3 text-sm leading-6 text-slate-300">
          {isSuccess
            ? 'Feel8 Studio should already be completing the sign-in flow in the background.'
            : 'The browser sign-in did not finish cleanly. Return to Feel8 Studio and try again if needed.'}
        </p>
        {!isSuccess && (error || errorDescription) ? (
          <p className="mt-4 text-sm text-rose-200">{errorDescription ? `${error}: ${errorDescription}` : error}</p>
        ) : null}
        {isSuccess ? (
          <p className="mt-4 text-sm text-slate-300">This page will return to the portal automatically in a moment.</p>
        ) : null}
        <div className="mt-8 flex flex-wrap gap-4 text-sm text-slate-300">
          <Link className="text-cyan-200 hover:text-white" to="/login">Open Portal</Link>
        </div>
      </div>
    </div>
  );
}
